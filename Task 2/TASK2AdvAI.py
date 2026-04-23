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
BATCH_SIZE = 64
EPOCHS = 14
N_FOLDS = 3
RANDOM_SEED = 42

GRADE_THRESHOLDS = {
    "A": {"color": 75, "size": 80, "ripeness": 70},
    "B": {"color": 65, "size": 70, "ripeness": 60},
}

CLASS_NAMES = []


def load_dataset_metadata(dataset_path: Path) -> pd.DataFrame:
    # Walk dataset directory and return a DataFrame with columns: filepath, label, category, condition
    print("debug statement - load_dataset_metadata")
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
    # Print class distribution and plot sample images per category
    print("debug statement - explore_dataset")
    os.makedirs("results", exist_ok=True)

    print("Null values per column:")
    print(df.isnull().sum())

    print("\nImage count per category and condition:")
    print(df.groupby(["category", "condition"]).size().reset_index(name="count").to_string(index=False))

    counts = df.groupby(["category", "condition"]).size().unstack()
    counts.plot(kind="bar", figsize=(14, 6), title="Image Count per Category")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("results/class_distribution.png")
    plt.show()


def build_data_generators(augment: bool = True) -> ImageDataGenerator:
    # Return an ImageDataGenerator — with augmentation for training, rescale only for validation
    print("debug statement - build_data_generators")
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


def preprocess_image(image_path: str) -> np.ndarray:
    # Load a single image, resize, normalise and return as array shaped (1, H, W, 3)
    print("debug statement - preprocess_image")
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = img_arr / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def build_cnn_model(num_classes: int, model_type: str = "cnn") -> tf.keras.Model:
    print("debug statement - build_cnn_model")
    if model_type == "resnet":
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
        )
        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax")
        ])

    else:
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2, 2),
            layers.Dropout(0.25),

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


def train_with_kfold(df: pd.DataFrame, num_classes: int, model_type: str = "cnn") -> list[dict]:
    # Run Stratified K-Fold training, return list of fold metrics
    print("debug statement - train_with_kfold")
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

        model = build_cnn_model(num_classes, model_type=model_type)

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
    print("debug statement - summarise_kfold_results")
    summary = pd.DataFrame([{
        "fold": r["fold"],
        "val_accuracy": r["val_accuracy"],
        "val_loss": r["val_loss"]
    } for r in fold_results])

    print("\n[INFO] K-Fold Summary:")
    print(summary.to_string(index=False))
    print(f"\nMean Accuracy : {summary['val_accuracy'].mean():.4f} ± {summary['val_accuracy'].std():.4f}")
    print(f"Mean Loss     : {summary['val_loss'].mean():.4f} ± {summary['val_loss'].std():.4f}")

    summary.plot(x="fold", y="val_accuracy", kind="bar", title="Accuracy per Fold", legend=False)
    plt.ylabel("Accuracy")
    plt.xlabel("Fold")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results/kfold_accuracy.png")
    plt.show()

    return summary


def evaluate_model(model: tf.keras.Model, test_generator) -> dict:
    # Evaluate model on test set, return accuracy, loss, classification report, confusion matrix
    print("debug statement - evaluate_model")
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\n[INFO] Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    report = classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys())
    print("\n[INFO] Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, list(test_generator.class_indices.keys()))

    return {
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "classification_report": report,
        "confusion_matrix": cm
    }


def plot_training_history(history, fold: int) -> None:
    # Plot and save accuracy and loss curves for a given fold
    print("debug statement - plot_training_history")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["accuracy"], label="Train Accuracy")
    ax1.plot(history["val_accuracy"], label="Val Accuracy")
    ax1.set_title(f"Fold {fold} - Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(history["loss"], label="Train Loss")
    ax2.plot(history["val_loss"], label="Val Loss")
    ax2.set_title(f"Fold {fold} - Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"results/fold_{fold}_history.png")
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    # Plot and save a labelled confusion matrix heatmap
    print("debug statement - plot_confusion_matrix")
    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()


def compute_quality_scores(model: tf.keras.Model, image_path: str) -> dict:
    # Run inference on an image and return color, size, ripeness scores, grade, and recommended action
    print("debug statement - compute_quality_scores")
    img_arr = preprocess_image(image_path)

    predictions = model.predict(img_arr, verbose=0)
    healthy_confidence = float(predictions[0][0]) * 100

    color = min(100, healthy_confidence * 1.0)
    size = min(100, healthy_confidence * 0.95)
    ripeness = min(100, healthy_confidence * 0.90)

    overall = round((color + size + ripeness) / 3, 2)
    grade, action = assign_grade(color, size, ripeness)

    return {
        "color": round(color, 2),
        "size": round(size, 2),
        "ripeness": round(ripeness, 2),
        "overall": overall,
        "grade": grade,
        "action": action
    }


def assign_grade(color: float, size: float, ripeness: float) -> tuple[str, str]:
    # Compare scores against thresholds and return (grade, action)
    print("debug statement - assign_grade")
    if color >= 75 and size >= 80 and ripeness >= 70:
        return "A", "Stock normally"
    elif color >= 65 and size >= 70 and ripeness >= 60:
        return "B", "Apply discount"
    else:
        return "C", "Remove from inventory"


def update_inventory(product_id: str, grade: str, quantity: int) -> dict:
    # Build and return an inventory update payload based on grade
    print("debug statement - update_inventory")
    discount = {"A": 0, "B": 20, "C": 100}.get(grade, 0)

    payload = {
        "product_id": product_id,
        "grade": grade,
        "quantity": quantity,
        "discount_pct": discount
    }

    print(f"[INFO] Inventory update: {payload}")

    # TODO: integrate with DESD API
    # requests.post("http://localhost:8000/api/inventory/update/", json=payload)

    return payload


def save_model(model: tf.keras.Model, path: str = "models/best_model.keras") -> None:
    # Save trained model to disk
    print("debug statement - save_model")
    os.makedirs("models", exist_ok=True)
    model.save(path)
    print(f"[INFO] Model saved to {path}")


def load_model(path: str = "models/best_model.keras") -> tf.keras.Model:
    # Load and return a saved model
    print("debug statement - load_model")
    model = tf.keras.models.load_model(path)
    print(f"[INFO] Model loaded from {path}")
    return model


def main():
    print("debug statement - main")
    df = load_dataset_metadata(DATASET_PATH)
    print(f"[INFO] Loaded {len(df)} images across {df['category'].nunique()} categories")

    explore_dataset(df)

    num_classes = df['label'].nunique()

    print("\n[INFO] Training CNN...")
    cnn_results, test_df = train_with_kfold(df, num_classes, model_type="cnn")
    cnn_summary = summarise_kfold_results(cnn_results)

    print("\n[INFO] Training ResNet50...")
    resnet_results, _ = train_with_kfold(df, num_classes, model_type="resnet")
    resnet_summary = summarise_kfold_results(resnet_results)

    print("\n[INFO] CNN Summary:")
    print(cnn_summary)
    print("\n[INFO] ResNet50 Summary:")
    print(resnet_summary)

    # TODO: load best model, evaluate on test set, run sample grading, save model
    print("[INFO] Pipeline complete.")


if __name__ == "__main__":
    main()