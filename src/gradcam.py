"""
PneumoScan — Grad-CAM Explainability
Gradient-weighted Class Activation Mapping for visual explanations.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from config import CLASS_NAMES, IMG_SIZE, GRADCAM_DIR
from utils import save_figure


def find_last_conv_layer(model):
    """Find the last convolutional layer in a model."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            # Nested model (e.g., ResNet base) — search inside
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name, layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, None
    return None, None


def generate_gradcam(model, image, class_index=None, layer_name=None):
    """
    Generate Grad-CAM heatmap for a given image and class.

    Args:
        model: Trained Keras model.
        image: Preprocessed image tensor (1, H, W, 3), values in [0, 1].
        class_index: Target class index. If None, uses predicted class.
        layer_name: Name of the conv layer to use. Auto-detected if None.

    Returns:
        heatmap: numpy array (H, W), values in [0, 1]
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    # Auto-detect last conv layer
    if layer_name is None:
        conv_name, parent_name = find_last_conv_layer(model)
        if conv_name is None:
            raise ValueError("No Conv2D layer found in model.")

        # Get the actual layer object
        if parent_name:
            parent = model.get_layer(parent_name)
            target_layer = parent.get_layer(conv_name)
        else:
            target_layer = model.get_layer(conv_name)
    else:
        target_layer = model.get_layer(layer_name)

    # Build a model that outputs both conv layer output and predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[target_layer.output, model.output],
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(image)
        if class_index is None:
            class_index = tf.argmax(predictions[0])
        loss = predictions[:, class_index]

    gradients = tape.gradient(loss, conv_output)

    # Pool gradients over spatial dimensions
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))

    # Weight the feature maps
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        image: Original image (H, W, 3), values in [0, 1] or [0, 255].
        heatmap: Grad-CAM heatmap (H, W), values in [0, 1].
        alpha: Overlay transparency.

    Returns:
        superimposed_img: Image with heatmap overlay (H, W, 3), values in [0, 1].
    """
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed.astype(np.float32) / 255.0


def generate_multiclass_gradcam(model, image):
    """
    Generate Grad-CAM heatmaps for all classes.

    Args:
        image: Preprocessed image (H, W, 3), values in [0, 1].

    Returns:
        dict: {class_name: (heatmap, overlay)}
    """
    results = {}
    for i, class_name in enumerate(CLASS_NAMES):
        heatmap = generate_gradcam(model, image, class_index=i)
        overlay = overlay_heatmap(image, heatmap)
        results[class_name] = (heatmap, overlay)
    return results


def plot_gradcam_grid(image, gradcam_results, model_name, save_path=None):
    """
    Plot: [Original | Normal-CAM | Bacteria-CAM | Virus-CAM]

    Args:
        image: Original image (H, W, 3).
        gradcam_results: Output of generate_multiclass_gradcam().
        model_name: For title.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original
    img_display = image if image.max() <= 1.0 else image / 255.0
    axes[0].imshow(img_display)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Per-class Grad-CAMs
    for idx, class_name in enumerate(CLASS_NAMES):
        _, overlay = gradcam_results[class_name]
        axes[idx + 1].imshow(overlay)
        axes[idx + 1].set_title(f"{class_name} CAM", fontsize=12, fontweight="bold")
        axes[idx + 1].axis("off")

    fig.suptitle(f"{model_name} — Grad-CAM Visualization", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def generate_gradcam_for_samples(model, dataset, model_name, n_per_class=5, save_dir=None):
    """Generate Grad-CAM visualizations for sample images from each class."""
    if save_dir is None:
        save_dir = os.path.join(GRADCAM_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    class_images = {i: [] for i in range(len(CLASS_NAMES))}

    for images, labels in dataset:
        for img, label in zip(images.numpy(), labels.numpy()):
            class_idx = np.argmax(label)
            if len(class_images[class_idx]) < n_per_class:
                class_images[class_idx].append(img)
        if all(len(v) >= n_per_class for v in class_images.values()):
            break

    for class_idx, class_name in enumerate(CLASS_NAMES):
        for i, img in enumerate(class_images[class_idx]):
            results = generate_multiclass_gradcam(model, img)
            save_path = os.path.join(save_dir, f"{class_name}_sample{i+1}.png")
            plot_gradcam_grid(img, results, model_name, save_path=save_path)

    print(f"Grad-CAM visualizations saved to: {save_dir}")
