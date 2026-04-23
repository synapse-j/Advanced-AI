"""
Task 2
"""
import kagglehub
path = kagglehub.dataset_download("muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Config
DATASET_PATH = Path(kagglehub.dataset_download("muhammad0subhan/fruit-and-vegetable-disease-healthy-vs-rotten"))
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 20
N_FOLDS = 5
RANDOM_SEED = 42

GRADE_THRESHOLDS = {
    "A": {"color": 75, "size": 80, "ripeness": 70},
    "B": {"color": 65, "size": 70, "ripeness": 60},
}

CLASS_NAMES = []


def load_dataset_metadata(dataset_path: Path) -> pd.DataFrame:

    data = []
    base_path = dataset_path / "Fruit And Vegetable Diseases Dataset"

    for category_folder in base_path.iterdir():
        if category_folder.is_dir():
            parts = category_folder.name.split("__")
            category = parts[0]
            condition = parts[1]
            label = 0 if condition.lower() == "healthy" else 1

            for image_file in category_folder.iterdir():
                if image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    data.append({
                        "filepath": str(image_file),
                        "label": label,
                        "category": category,
                        "condition": condition
                    })

    df = pd.DataFrame(data)
    print(df.head(2))
    return df


def explore_dataset(df: pd.DataFrame) -> None:
    os.makedirs("results", exist_ok=True)
    print("Null values per column:")
    print(df.isnull().sum())

    # class distribution
    print("\nImage count per category and condition:")
    print(df.groupby(["category", "condition"]).size().reset_index(name="count").to_string(index=False))

    # bar chart
    counts = df.groupby(["category", "condition"]).size().unstack()
    counts.plot(kind="bar", figsize=(14, 6), title="Image Count per Category")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png")
    plt.show()
    '''' Code not needed for normal epoc, used to intially check for potential image corruption
    print("\nCorrupted Image Check")
    corrupt = []
    for filepath in df["filepath"]:
        try:
            img = tf.keras.preprocessing.image.load_img(filepath)
        except Exception:
            corrupt.append(filepath)
    print(f"[INFO] Corrupt images found: {len(corrupt)}")
    pass
    '''


def build_data_generators(augment: bool = True) -> ImageDataGenerator:
    if augment:
        return ImageDataGenerator(
            rescale=1./255,
            horizontal_flip=True,
            vertical_flip=True,
        )
    else:
        
        return ImageDataGenerator(
            rescale=1./255
        )
    pass


def preprocess_image(image_path: str) -> np.ndarray:
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def build_cnn_model(num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        # 1
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),

        # Connected
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_with_kfold(df: pd.DataFrame, num_classes: int) -> list[dict]:

    os.makedirs("models", exist_ok=True)

    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED, stratify=df["label"])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df["filepath"], train_df["label"]), 1):
        print(f"\n[INFO] Training Fold {fold}/{N_FOLDS}")

        fold_train = train_df.iloc[train_idx]
        fold_val = train_df.iloc[val_idx]

        train_gen = build_data_generators(augment=True)
        val_gen = build_data_generators(augment=False)

        train_data = train_gen.flow_from_dataframe(
            fold_train, x_col="filepath", y_col="condition",
            target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="categorical"
        )
        val_data = val_gen.flow_from_dataframe(
            fold_val, x_col="filepath", y_col="condition",
            target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="categorical"
        )

        model = build_cnn_model(num_classes)

        early_stop = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        checkpoint = callbacks.ModelCheckpoint(
            f"models/fold_{fold}.keras", save_best_only=True, monitor="val_loss"
        )

        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=EPOCHS,
            callbacks=[early_stop, checkpoint]
        )

        val_loss, val_acc = model.evaluate(val_data)
        print(f"[INFO] Fold {fold} — Val Accuracy: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        fold_results.append({
            "fold": fold,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "history": history.history
        })

    return fold_results, test_df


def summarise_kfold_results(fold_results: list[dict]) -> pd.DataFrame:
    # Return a DataFrame summarising mean and std of accuracy and loss across folds
    pass


def evaluate_model(model: tf.keras.Model, test_generator) -> dict:
    # Evaluate model on test set, return accuracy, loss, classification report, confusion matrix
    pass


def plot_training_history(history, fold: int) -> None:
    # Plot and save accuracy and loss curves for a given fold
    pass


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    # Plot and save a labelled confusion matrix heatmap
    pass


def compute_quality_scores(model: tf.keras.Model, image_path: str) -> dict:
    # Run inference on an image and return color, size, ripeness scores, grade, and recommended action
    pass


def assign_grade(color: float, size: float, ripeness: float) -> tuple[str, str]:
    # Compare scores against thresholds and return (grade, action)
    pass


def update_inventory(product_id: str, grade: str, quantity: int) -> dict:
    # Build and return an inventory update payload based on grade
    pass


def save_model(model: tf.keras.Model, path: str = "models/best_model.keras") -> None:
    # Save trained model to disk
    pass


def load_model(path: str = "models/best_model.keras") -> tf.keras.Model:
    # Load and return a saved model
    pass


def main():
    df = load_dataset_metadata(DATASET_PATH)
    print(f"[INFO] Loaded {len(df)} images across {df['category'].nunique()} categories")

    explore_dataset(df)

    num_classes = df['label'].nunique()
    fold_results = train_with_kfold(df, num_classes)

    summary = summarise_kfold_results(fold_results)
    print(summary)

    # TODO: load best model, evaluate on test set, run sample grading, save model
    print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()