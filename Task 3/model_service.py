"""
AI Acknowledgement: Parts of this code were developed with assistance from AI tools
including Claude (Anthropic). All AI-generated content has been reviewed, understood,
and verified by the authors. Use of AI tools is declared in accordance with UWE
Bristol's academic integrity guidelines for module UFCFUR-15-3.

Task 3 + Task 4 — Model Service with GRAD-CAM Explainability
Module: UFCFUR-15-3 Advanced Artificial Intelligence
Institution: UWE Bristol, 2025-26

This module provides a single public function, classify_produce(), that the DESD
application calls to classify an image of produce.  It is the only integration
point between the standalone training pipeline (train.py) and the DESD application.

--- What is GRAD-CAM? (Task 4 — Explainable AI requirement) ---
GRAD-CAM (Gradient-weighted Class Activation Mapping) is an explainability technique
for convolutional neural networks.  It answers the question: "which pixels in this
image most influenced the model's decision?"

How it works:
  1. A GradientTape records the forward pass through the model.
  2. The gradient of the predicted class score with respect to the output of the
     last convolutional layer is computed.
  3. These gradients are globally average-pooled to produce a weight vector — one
     weight per feature map channel.
  4. The feature maps are combined using these weights and passed through ReLU
     (keeping only activations that positively influenced the prediction).
  5. The resulting heatmap is up-sampled to the input image size and overlaid as a
     colour gradient on the original image.

Why use GRAD-CAM here?
  For a food safety application it is not enough to know whether produce is fresh;
  we also need to understand WHY the model made that decision.  GRAD-CAM lets a
  producer or marketplace operator see which region (discolouration, bruise, mould
  patch) triggered the "rotten" prediction, making the system auditable and
  trustworthy — a core XAI requirement for the BRFN platform.

Dependencies (install before use):
    pip install tensorflow opencv-python numpy
"""

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Paths — resolved relative to this file so the service works from any cwd
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent
_MODEL_PATH = _THIS_DIR / "freshness_model.h5"
_CLASS_NAMES_PATH = _THIS_DIR / "class_names.json"

IMG_HEIGHT = 224
IMG_WIDTH = 224


# ---------------------------------------------------------------------------
# Lazy model loading (singleton — loaded once, reused for every request)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> tf.keras.Model:
    """Load the freshness model from disk (cached after first call).

    Returns:
        Loaded tf.keras.Model.

    Raises:
        FileNotFoundError: If freshness_model.h5 has not been created by train.py.
    """
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {_MODEL_PATH}.\n"
            "Run train.py first to train and save the model."
        )
    model = tf.keras.models.load_model(str(_MODEL_PATH))
    print(f"[model_service] Model loaded ← {_MODEL_PATH}")
    return model


@lru_cache(maxsize=1)
def _load_class_names() -> list:
    """Load the ordered class name list saved by train.py (cached after first call).

    Returns:
        List of class name strings in the same order as the model's output layer.

    Raises:
        FileNotFoundError: If class_names.json has not been created by train.py.
    """
    if not _CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names not found at {_CLASS_NAMES_PATH}.\n"
            "Run train.py first to train and save the model."
        )
    with open(_CLASS_NAMES_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _preprocess(image_path: str) -> np.ndarray:
    """Load and normalise an image for model inference.

    Args:
        image_path: Path to the image file (JPEG or PNG).

    Returns:
        Float32 array of shape (1, IMG_HEIGHT, IMG_WIDTH, 3) in [0, 1].
    """
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)


# ---------------------------------------------------------------------------
# GRAD-CAM (Task 4 — Explainable AI)
# ---------------------------------------------------------------------------

def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    """Locate the last Conv2D layer in the model for GRAD-CAM.

    Because train.py builds the model using base_model.input / base_model.output
    (flat functional API), all MobileNetV2 Conv2D layers appear directly in
    model.layers — no nested sub-model traversal is needed.

    Args:
        model: The loaded freshness model.

    Returns:
        The last tf.keras.layers.Conv2D layer in the model.

    Raises:
        ValueError: If no Conv2D layer is found.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError(
        "No Conv2D layer found in the model. "
        "Ensure the model was built with the flat functional API (base_model.input / output)."
    )


def _compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_array: np.ndarray,
    pred_index: int,
) -> np.ndarray:
    """Compute a GRAD-CAM heatmap for a given prediction.

    Implements Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization".

    Steps:
        1. Build a sub-model from the input to (last_conv_output, final_predictions).
        2. Record gradients of the top class score w.r.t. the last conv output.
        3. Global-average-pool the gradients to get per-channel weights.
        4. Compute a weighted sum of the feature maps, apply ReLU, and normalise.

    Args:
        model: Loaded Keras model with flat layer structure.
        img_array: Pre-processed image array of shape (1, H, W, 3).
        pred_index: Integer index of the class whose activation map to visualise.

    Returns:
        2-D float32 heatmap array with values in [0, 1], shape (H_conv, W_conv).
    """
    last_conv = _find_last_conv_layer(model)

    # Sub-model: input → (last conv feature maps, final softmax output)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv.output, model.output],
    )

    with tf.GradientTape() as tape:
        # Cast to float32 so tape can differentiate
        img_tensor = tf.cast(img_array, tf.float32)
        conv_output, predictions = grad_model(img_tensor, training=False)
        class_score = predictions[:, pred_index]

    # Gradient of class score w.r.t. last conv layer output
    grads = tape.gradient(class_score, conv_output)  # (1, H, W, C)

    # Average over spatial dimensions → one importance weight per channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight each feature map channel and sum
    conv_output = conv_output[0]                                   # (H, W, C)
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]          # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                                  # (H, W)

    # ReLU removes negative activations; normalise to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()


def _overlay_heatmap(image_path: str, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Resize the GRAD-CAM heatmap and overlay it on the original image.

    The heatmap is colourised using OpenCV's COLORMAP_JET (blue=low, red=high)
    and blended with the original image so the user can see which regions
    contributed most to the model's prediction.

    Args:
        image_path: Path to the original input image.
        heatmap: 2-D float32 heatmap from _compute_gradcam_heatmap.
        alpha: Blend weight for the heatmap overlay (0 = no overlay, 1 = full).

    Returns:
        BGR uint8 numpy array of the overlay image, shape (IMG_HEIGHT, IMG_WIDTH, 3).
    """
    img_bgr = cv2.imread(image_path)
    img_bgr = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT))

    heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT)))
    heatmap_coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Weighted blend: original image retains (1 - alpha) of its brightness
    overlay = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap_coloured, alpha, 0)
    return overlay


def _encode_image_base64(img_bgr: np.ndarray) -> str:
    """Encode a BGR OpenCV image as a base64 PNG string.

    The DESD frontend can embed this string directly in an <img src="data:..."> tag.

    Args:
        img_bgr: BGR uint8 image array.

    Returns:
        Base64-encoded PNG string (UTF-8 decoded, no line breaks).
    """
    success, buffer = cv2.imencode(".png", img_bgr)
    if not success:
        raise RuntimeError("cv2.imencode failed when encoding GRAD-CAM overlay.")
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Plain-English explanation
# ---------------------------------------------------------------------------

def _build_explanation(
    predicted_class: str,
    confidence: float,
    is_fresh: bool,
) -> str:
    """Generate a plain-English explanation of the model's prediction.

    Intended for display to non-technical users on the BRFN marketplace.

    Args:
        predicted_class: Raw class name string (e.g. "Apple__Healthy").
        confidence: Prediction confidence as a percentage (0–100).
        is_fresh: True if the predicted class is a healthy/fresh class.

    Returns:
        Human-readable explanation string.
    """
    parts = predicted_class.split("__")
    produce_type = parts[0] if len(parts) == 2 else predicted_class
    condition_str = "fresh and suitable for sale" if is_fresh else "rotten and unsuitable for sale"

    freshness_advice = (
        "It has been added to the inventory as available stock."
        if is_fresh
        else "It has been flagged for removal from the inventory."
    )

    return (
        f"The AI model has identified this produce as {produce_type} and classified it as "
        f"{condition_str} with {confidence:.1f}% confidence. "
        f"{freshness_advice} "
        f"The GRAD-CAM heatmap highlights the image regions — such as discolouration, "
        f"bruising, or mould — that most influenced this classification decision."
    )


# ---------------------------------------------------------------------------
# Public API — single integration point between DESD and the ML model
# ---------------------------------------------------------------------------

def classify_produce(image_path: str) -> dict[str, Any]:
    """Classify a produce image as fresh or rotten and return an XAI explanation.

    This is the ONLY function that the DESD application needs to call.  It loads
    the model lazily on first use and keeps it in memory for subsequent calls.

    Args:
        image_path: Absolute or relative path to a JPEG or PNG image of produce.

    Returns:
        Dict with the following keys:

        predicted_class (str):
            The specific class predicted, e.g. "Apple__Healthy" or "Banana__Rotten".

        confidence (float):
            Prediction confidence as a percentage in the range 0–100.

        is_fresh (bool):
            True if the predicted class is a "Healthy" class; False if "Rotten".

        all_probabilities (dict):
            Mapping of every class name to its softmax probability (0–1).

        gradcam_base64 (str):
            Base64-encoded PNG of the GRAD-CAM heatmap overlaid on the input image.
            Embed in HTML as:  <img src="data:image/png;base64,{gradcam_base64}" />

        explanation (str):
            A plain-English description of the prediction and its implications for
            inventory management, suitable for display to end users.
    """
    model = _load_model()
    class_names = _load_class_names()

    # Pre-process image for inference
    img_array = _preprocess(image_path)

    # Run inference
    raw_probs = model.predict(img_array, verbose=0)[0]         # shape: (num_classes,)
    pred_index = int(np.argmax(raw_probs))
    predicted_class = class_names[pred_index]
    confidence = float(raw_probs[pred_index]) * 100.0
    is_fresh = "healthy" in predicted_class.lower()

    # Build full probability dict for all classes
    all_probabilities = {
        name: round(float(prob), 6)
        for name, prob in zip(class_names, raw_probs)
    }

    # --- GRAD-CAM (Task 4 — Explainable AI) ---
    # Compute the gradient-weighted class activation heatmap so the user can see
    # WHICH regions of the image drove the fresh/rotten decision.
    heatmap = _compute_gradcam_heatmap(model, img_array, pred_index)
    overlay = _overlay_heatmap(image_path, heatmap)
    gradcam_base64 = _encode_image_base64(overlay)

    # Plain-English explanation for the BRFN marketplace UI
    explanation = _build_explanation(predicted_class, confidence, is_fresh)

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "is_fresh": is_fresh,
        "all_probabilities": all_probabilities,
        "gradcam_base64": gradcam_base64,
        "explanation": explanation,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test (run this file directly: python model_service.py <image_path>)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python model_service.py <path_to_image>")
        sys.exit(1)

    result = classify_produce(sys.argv[1])
    print("\n--- classify_produce() result ---")
    print(f"  predicted_class : {result['predicted_class']}")
    print(f"  confidence      : {result['confidence']:.2f}%")
    print(f"  is_fresh        : {result['is_fresh']}")
    print(f"  explanation     : {result['explanation']}")
    print(f"  gradcam_base64  : <{len(result['gradcam_base64'])} chars>")
    print(f"  all_probabilities (top 5):")
    top5 = sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True)[:5]
    for cls, prob in top5:
        print(f"    {cls}: {prob:.4f}")
