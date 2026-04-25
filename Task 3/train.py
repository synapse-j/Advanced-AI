"""
AI Acknowledgement: Parts of this code were developed with assistance from AI tools
including Claude (Anthropic). All AI-generated content has been reviewed, understood,
and verified by the authors. 

Task 3 — Standalone Model Training

This file is COMPLETELY STANDALONE.  It has zero dependency on the DESD application
or any other part of the project.  It reads images from the dataset/ folder sitting
next to this file, trains a MobileNetV2 freshness classifier, evaluates it, and
writes two artefacts consumed by model_service.py:

    freshness_model.h5   — the saved Keras model
    class_names.json     — ordered list of class names matching the model's output layer

Dataset layout expected inside dataset/:
    dataset/
    └── Fruit And Vegetable Diseases Dataset/
        ├── Apple__Healthy/
        ├── Apple__Rotten/
        ├── Banana__Healthy/
        ...

Training uses two-phase transfer learning:
    Phase 1 — MobileNetV2 base frozen; only the Dense classification head is trained.
    Phase 2 — Top FINE_TUNE_AT+ layers of MobileNetV2 are unfrozen and fine-tuned at
               a lower learning rate (1e-5) to improve per-class accuracy.
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------------------------------------------------------------------------
# Configuration — edit these paths/hyper-parameters if needed
# ---------------------------------------------------------------------------

# Paths are relative to this file so the module is truly standalone
THIS_DIR = Path(__file__).resolve().parent
DATASET_ROOT = THIS_DIR / "dataset" / "Fruit And Vegetable Diseases Dataset"
MODEL_SAVE_PATH = THIS_DIR / "freshness_model.h5"
CLASS_NAMES_PATH = THIS_DIR / "class_names.json"
RESULTS_DIR = THIS_DIR / "results"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
PHASE1_EPOCHS = 10   # epochs with base frozen
PHASE2_EPOCHS = 10   # epochs fine-tuning unfrozen top layers
FINE_TUNE_AT = 100   # freeze MobileNetV2 layers before this index in phase 2
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(dataset_root: Path) -> pd.DataFrame:
    """Walk the dataset folder and build a DataFrame of image paths and labels.

    Each sub-folder is named <Produce>__<Condition> (e.g. Apple__Healthy).
    The folder name becomes the class_name used by the generator.

    Args:
        dataset_root: Path to the folder containing the class sub-folders.

    Returns:
        DataFrame with columns: filepath (str), class_name (str), label (int).
    """
    records = []
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_root}.\n"
            "Please download and extract the Kaggle dataset so that\n"
            "dataset/Fruit And Vegetable Diseases Dataset/<Class>/ folders exist\n"
            "next to train.py."
        )

    for folder in sorted(dataset_root.iterdir()):
        if not folder.is_dir():
            continue
        parts = folder.name.split("__")
        if len(parts) != 2:
            continue
        condition = parts[1]
        label = 0 if condition.lower() == "healthy" else 1
        for img in folder.iterdir():
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                records.append(
                    {"filepath": str(img), "class_name": folder.name, "label": label}
                )

    df = pd.DataFrame(records)
    print(
        f"[INFO] Dataset loaded: {len(df)} images | "
        f"{df['class_name'].nunique()} classes"
    )
    return df


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def make_generator(augment: bool = False) -> ImageDataGenerator:
    """Create an ImageDataGenerator with optional data augmentation.

    Augmentation is used only for the training split to prevent overfitting.

    Args:
        augment: Apply random flips, rotation, and zoom when True.

    Returns:
        Configured ImageDataGenerator.
    """
    if augment:
        return ImageDataGenerator(
            rescale=1.0 / 255,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.1,
            height_shift_range=0.1,
        )
    return ImageDataGenerator(rescale=1.0 / 255)


# ---------------------------------------------------------------------------
# Model — MobileNetV2 transfer learning (flat functional API)
# ---------------------------------------------------------------------------

def build_model(num_classes: int):
    """Construct a MobileNetV2 transfer learning model.

    The model is built using base_model.input / base_model.output so that all
    MobileNetV2 layers appear flat in model.layers.  This is required for the
    GRAD-CAM implementation in model_service.py to locate the last Conv2D layer
    without traversing a nested sub-model.

    Args:
        num_classes: Number of output classes.

    Returns:
        Tuple of (compiled tf.keras.Model, MobileNetV2 base model reference).
    """
    base = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # phase 1: freeze base

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ---------------------------------------------------------------------------
# Two-phase training
# ---------------------------------------------------------------------------

def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame,
                sorted_classes: list, num_classes: int) -> tuple:
    """Train a MobileNetV2 model using two-phase transfer learning.

    Phase 1 trains only the Dense head with the MobileNetV2 base frozen.
    Phase 2 unfreezes the top FINE_TUNE_AT+ layers and fine-tunes them at
    a lower learning rate to adapt ImageNet features to produce imagery.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        sorted_classes: Alphabetically sorted list of class names.
        num_classes: Total number of output classes.

    Returns:
        Tuple of (trained tf.keras.Model, combined training history dict).
    """
    train_gen = make_generator(augment=True)
    val_gen = make_generator(augment=False)

    train_data = train_gen.flow_from_dataframe(
        train_df,
        x_col="filepath",
        y_col="class_name",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=sorted_classes,
    )
    val_data = val_gen.flow_from_dataframe(
        val_df,
        x_col="filepath",
        y_col="class_name",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=sorted_classes,
    )

    model, base = build_model(num_classes)

    # ---- Phase 1: train Dense head (base frozen) ----
    print("[INFO] Phase 1 — training classification head (MobileNetV2 base frozen)")
    checkpoint = callbacks.ModelCheckpoint(
        str(MODEL_SAVE_PATH),
        save_best_only=True,
        monitor="val_loss",
    )
    early1 = callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss")
    history1 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=PHASE1_EPOCHS,
        callbacks=[early1, checkpoint],
        verbose=1,
    )

    # ---- Phase 2: fine-tune top layers of MobileNetV2 ----
    print("[INFO] Phase 2 — fine-tuning top MobileNetV2 layers (lr=1e-5)")
    base.trainable = True
    for lyr in base.layers[:FINE_TUNE_AT]:
        lyr.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    early2 = callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss")
    history2 = model.fit(
        train_data,
        validation_data=val_data,
        epochs=PHASE2_EPOCHS,
        callbacks=[early2, checkpoint],
        verbose=1,
    )

    # Merge histories into one continuous record
    combined = {
        key: history1.history.get(key, []) + history2.history.get(key, [])
        for key in set(history1.history) | set(history2.history)
    }
    return model, combined


# ---------------------------------------------------------------------------
# Evaluation and visualisation
# ---------------------------------------------------------------------------

def evaluate(model: tf.keras.Model, test_df: pd.DataFrame,
             sorted_classes: list) -> None:
    """Run evaluation on the held-out test set and print / save all metrics.

    Metrics reported: accuracy, weighted precision, recall, F1, per-class
    classification report, and a confusion matrix heatmap PNG.

    Args:
        model: Trained Keras model.
        test_df: Test split DataFrame with filepath and class_name columns.
        sorted_classes: Alphabetically sorted list of class names.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    class_indices = {name: i for i, name in enumerate(sorted_classes)}

    test_gen = make_generator(augment=False)
    test_data = test_gen.flow_from_dataframe(
        test_df,
        x_col="filepath",
        y_col="class_name",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        classes=sorted_classes,
    )

    preds = model.predict(test_data, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = [class_indices[cn] for cn in test_df["class_name"]]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=sorted_classes, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n[RESULTS] Accuracy:  {acc:.4f}")
    print(f"[RESULTS] Precision: {prec:.4f}")
    print(f"[RESULTS] Recall:    {rec:.4f}")
    print(f"[RESULTS] F1 Score:  {f1:.4f}")
    print("\n[RESULTS] Classification Report:")
    print(report)

    # Save classification report
    report_path = RESULTS_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"[INFO] Report saved → {report_path}")

    # Confusion matrix heatmap
    fig_size = max(10, len(sorted_classes))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=sorted_classes,
        yticklabels=sorted_classes,
    )
    plt.title("Confusion Matrix — Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = RESULTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved → {cm_path}")


def plot_history(history: dict) -> None:
    """Plot and save the combined phase-1 + phase-2 training curves.

    Args:
        history: Dict with accuracy, val_accuracy, loss, val_loss lists.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History — Phase 1 + Phase 2", fontsize=13)

    ax1.plot(epochs_range, history["accuracy"], label="Train")
    ax1.plot(epochs_range, history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    ax2.plot(epochs_range, history["loss"], label="Train")
    ax2.plot(epochs_range, history["val_loss"], label="Validation")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    out_path = RESULTS_DIR / "training_history.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Training history saved → {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete standalone training pipeline.

    Steps:
        1. Load image metadata from dataset/
        2. Split into train / validation / test (70 / 15 / 15)
        3. Two-phase MobileNetV2 training
        4. Evaluate on test set (accuracy, precision, recall, F1, confusion matrix)
        5. Save freshness_model.h5 and class_names.json
    """
    df = load_data(DATASET_ROOT)
    sorted_classes = sorted(df["class_name"].unique())
    num_classes = len(sorted_classes)

    # 70 / 15 / 15 split stratified by binary label
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["label"]
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(
        f"[INFO] Split — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}"
    )

    model, history = train_model(train_df, val_df, sorted_classes, num_classes)
    plot_history(history)

    evaluate(model, test_df, sorted_classes)

    # Save the trained model
    model.save(str(MODEL_SAVE_PATH))
    print(f"[INFO] Model saved → {MODEL_SAVE_PATH}")

    # Save class names so model_service.py can reconstruct predictions
    with open(CLASS_NAMES_PATH, "w", encoding="utf-8") as fh:
        json.dump(sorted_classes, fh, indent=2)
    print(f"[INFO] Class names saved → {CLASS_NAMES_PATH}")

    print("\n[INFO] Standalone training complete.")
    print(f"       Model artefacts written to: {THIS_DIR}")


if __name__ == "__main__":
    main()
