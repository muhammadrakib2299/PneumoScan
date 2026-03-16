"""
PneumoScan — Gradio Web Demo
Upload a chest X-ray → Get prediction + Grad-CAM heatmap + confidence scores.

Run:
    python app/demo.py
"""

import os
import sys
import numpy as np
import cv2
import gradio as gr
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use("Agg")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import CLASS_NAMES, IMG_SIZE, MODEL_SAVE_PATHS, ENSEMBLE_CONFIG_PATH
from gradcam import generate_gradcam, overlay_heatmap


def load_available_models():
    """Load all available trained models."""
    models = {}
    for name, path in MODEL_SAVE_PATHS.items():
        if os.path.exists(path):
            try:
                models[name] = keras.models.load_model(path)
                print(f"Loaded: {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
    return models


def preprocess_image(image):
    """Preprocess uploaded image for model inference."""
    if image is None:
        return None
    img = cv2.resize(image, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return img


def predict(image, model_name):
    """Run prediction on uploaded image."""
    if image is None:
        return None, "Please upload an image.", {}

    if model_name not in MODELS:
        return None, f"Model '{model_name}' not loaded.", {}

    model = MODELS[model_name]
    img = preprocess_image(image)
    img_batch = np.expand_dims(img, axis=0)

    # Predict
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])

    # Confidence dict for Gradio label
    confidences = {name: float(prob) for name, prob in zip(CLASS_NAMES, predictions)}

    # Grad-CAM
    try:
        heatmap = generate_gradcam(model, img, class_index=predicted_idx)
        gradcam_img = overlay_heatmap(img, heatmap, alpha=0.4)
        gradcam_img = (gradcam_img * 255).astype(np.uint8)
    except Exception:
        gradcam_img = (img * 255).astype(np.uint8)

    result_text = f"**{predicted_class}** — Confidence: {confidence:.1%}"

    return gradcam_img, result_text, confidences


# Load models at startup
print("Loading models...")
MODELS = load_available_models()

if not MODELS:
    print("Warning: No trained models found. Train models first.")
    AVAILABLE_MODELS = ["No models available"]
else:
    AVAILABLE_MODELS = list(MODELS.keys())
    print(f"Available models: {AVAILABLE_MODELS}")


# Build Gradio interface
with gr.Blocks(
    title="PneumoScan",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # PneumoScan
        ### Multi-Class Pneumonia Detection from Chest X-Rays

        Upload a chest X-ray image to get:
        - **Predicted class** (Normal / Bacterial Pneumonia / Viral Pneumonia)
        - **Confidence scores** for each class
        - **Grad-CAM heatmap** showing which regions the model focuses on
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Chest X-Ray", type="numpy")
            model_dropdown = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0] if AVAILABLE_MODELS else None,
                label="Select Model",
            )
            predict_btn = gr.Button("Analyze X-Ray", variant="primary")

        with gr.Column():
            gradcam_output = gr.Image(label="Grad-CAM Heatmap")
            result_text = gr.Markdown(label="Prediction")
            confidence_output = gr.Label(label="Confidence Scores", num_top_classes=3)

    predict_btn.click(
        fn=predict,
        inputs=[image_input, model_dropdown],
        outputs=[gradcam_output, result_text, confidence_output],
    )

    gr.Markdown(
        """
        ---
        **Disclaimer**: This tool is for educational and research purposes only.
        It is not a substitute for professional medical diagnosis.

        Built with TensorFlow, Grad-CAM, and Gradio.
        """
    )

if __name__ == "__main__":
    demo.launch(share=False)
