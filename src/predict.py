"""
PneumoScan — Single Image Inference
Predict class + confidence for any chest X-ray image with Grad-CAM output.

Usage:
    python src/predict.py --image path/to/xray.jpg --model resnet50
    python src/predict.py --image path/to/xray.jpg --model best
"""

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from config import (
    CLASS_NAMES, IMG_SIZE, MODEL_SAVE_PATHS, MODELS_DIR,
    ENSEMBLE_CONFIG_PATH, GRADCAM_DIR,
)
from gradcam import generate_gradcam, overlay_heatmap
from utils import save_figure


def load_and_preprocess_image(image_path):
    """Load and preprocess a single image for inference."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized, img_resized


def predict_single_image(model, image):
    """
    Predict class and confidence for a single image.

    Args:
        model: Trained Keras model.
        image: Preprocessed image (H, W, 3), values in [0, 1].

    Returns:
        predicted_class: str
        confidence: float
        probabilities: dict {class_name: probability}
    """
    img_batch = np.expand_dims(image, axis=0)
    predictions = model.predict(img_batch, verbose=0)[0]

    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])

    probabilities = {name: float(prob) for name, prob in zip(CLASS_NAMES, predictions)}

    return predicted_class, confidence, probabilities


def predict_with_explanation(model, image_path, model_name="model", save_dir=None):
    """
    Full prediction pipeline: predict + Grad-CAM + visualization.

    Args:
        model: Trained Keras model.
        image_path: Path to chest X-ray image.
        model_name: Model name for titles.
        save_dir: Directory to save outputs. If None, displays plots.

    Returns:
        result dict with prediction, confidence, probabilities, gradcam paths
    """
    # Load and preprocess
    img_normalized, img_original = load_and_preprocess_image(image_path)

    # Predict
    predicted_class, confidence, probabilities = predict_single_image(model, img_normalized)

    # Print results
    print(f"\n{'=' * 40}")
    print(f"PneumoScan Prediction ({model_name})")
    print(f"{'=' * 40}")
    print(f"Image: {image_path}")
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nClass Probabilities:")
    for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"  {cls:12s} {prob:.4f} {bar}")

    # Grad-CAM
    predicted_idx = np.argmax(list(probabilities.values()))
    heatmap = generate_gradcam(model, img_normalized, class_index=predicted_idx)
    gradcam_overlay = overlay_heatmap(img_normalized, heatmap)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(img_normalized)
    axes[0].set_title("Original X-Ray", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Grad-CAM overlay
    axes[1].imshow(gradcam_overlay)
    axes[1].set_title(f"Grad-CAM ({predicted_class})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Confidence bar chart
    classes = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = ["#2196F3" if c != predicted_class else "#F44336" for c in classes]
    axes[2].barh(classes, probs, color=colors, edgecolor="black")
    axes[2].set_xlim(0, 1)
    axes[2].set_title("Confidence Scores", fontsize=12, fontweight="bold")
    for i, prob in enumerate(probs):
        axes[2].text(prob + 0.02, i, f"{prob:.2%}", va="center", fontsize=10)

    fig.suptitle(f"PneumoScan — {predicted_class} ({confidence:.1%})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.splitext(os.path.basename(image_path))[0]
        save_path = os.path.join(save_dir, f"{filename}_prediction.png")
        save_figure(fig, save_path)
    else:
        plt.show()

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def get_best_model_path():
    """Get the path to the best performing model (from ensemble config or default)."""
    if os.path.exists(ENSEMBLE_CONFIG_PATH):
        import json
        with open(ENSEMBLE_CONFIG_PATH) as f:
            config = json.load(f)
        best_name = config["models"][0]
        return MODEL_SAVE_PATHS.get(best_name), best_name

    # Fallback: return first available model
    for name, path in MODEL_SAVE_PATHS.items():
        if os.path.exists(path):
            return path, name

    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PneumoScan — Single Image Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to chest X-ray image")
    parser.add_argument("--model", type=str, default="best", help="Model name or 'best'")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save output")
    args = parser.parse_args()

    # Load model
    if args.model == "best":
        model_path, model_name = get_best_model_path()
        if model_path is None:
            print("Error: No trained models found. Train a model first.")
            exit(1)
    else:
        model_name = args.model
        model_path = MODEL_SAVE_PATHS.get(model_name)

    if not model_path or not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit(1)

    print(f"Loading model: {model_name} from {model_path}")
    model = keras.models.load_model(model_path)

    predict_with_explanation(model, args.image, model_name, save_dir=args.save_dir)
