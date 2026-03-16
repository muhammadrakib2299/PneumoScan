"""
PneumoScan — LIME Explainability
Local Interpretable Model-agnostic Explanations for chest X-ray predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from config import CLASS_NAMES, IMG_SIZE, LIME_DIR
from utils import save_figure


def create_lime_explainer():
    """Create a LIME image explainer instance."""
    return lime_image.LimeImageExplainer(random_state=42)


def explain_image(model, image, explainer=None, num_samples=1000, num_features=10):
    """
    Generate LIME explanation for a single image.

    Args:
        model: Trained Keras model.
        image: Preprocessed image (H, W, 3), values in [0, 1].
        explainer: LIME explainer instance. Created if None.
        num_samples: Number of perturbed samples for LIME.
        num_features: Number of superpixel features to show.

    Returns:
        explanation: LIME explanation object.
    """
    if explainer is None:
        explainer = create_lime_explainer()

    if image.ndim == 4:
        image = image[0]

    # LIME expects images in [0, 1]
    if image.max() > 1.0:
        image = image / 255.0

    def predict_fn(images):
        """Prediction function for LIME — expects batch of images."""
        images = np.array(images, dtype=np.float32)
        if images.max() > 1.0:
            images = images / 255.0
        return model.predict(images, verbose=0)

    explanation = explainer.explain_instance(
        image.astype(np.float64),
        predict_fn,
        top_labels=len(CLASS_NAMES),
        hide_color=0,
        num_samples=num_samples,
        num_features=num_features,
    )

    return explanation


def plot_lime_explanation(image, explanation, model_name, predicted_class=None, save_path=None):
    """
    Plot LIME explanation showing positive/negative contributions per class.

    Args:
        image: Original image (H, W, 3), values in [0, 1].
        explanation: LIME explanation object.
        model_name: For title.
        predicted_class: Index of predicted class.
    """
    if image.max() > 1.0:
        image = image / 255.0

    n_classes = len(CLASS_NAMES)
    fig, axes = plt.subplots(1, n_classes + 1, figsize=(5 * (n_classes + 1), 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Per-class LIME explanations
    for i, class_name in enumerate(CLASS_NAMES):
        try:
            temp, mask = explanation.get_image_and_mask(
                i,
                positive_only=False,
                num_features=10,
                hide_rest=False,
            )
            bounded = mark_boundaries(temp, mask, color=(1, 0, 0), mode="thick")
            axes[i + 1].imshow(bounded)

            title = f"{class_name} LIME"
            if predicted_class is not None and i == predicted_class:
                title += " (predicted)"
            axes[i + 1].set_title(title, fontsize=12, fontweight="bold")
        except Exception:
            axes[i + 1].imshow(image)
            axes[i + 1].set_title(f"{class_name} (N/A)", fontsize=12)
        axes[i + 1].axis("off")

    fig.suptitle(f"{model_name} — LIME Explanation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def plot_lime_positive_negative(image, explanation, class_index, model_name, save_path=None):
    """Plot separate positive and negative LIME contributions."""
    if image.max() > 1.0:
        image = image / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Positive contributions
    temp_pos, mask_pos = explanation.get_image_and_mask(
        class_index, positive_only=True, num_features=10, hide_rest=True,
    )
    axes[1].imshow(mark_boundaries(temp_pos, mask_pos, color=(0, 1, 0)))
    axes[1].set_title(f"{CLASS_NAMES[class_index]} — Positive", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Negative contributions
    temp_neg, mask_neg = explanation.get_image_and_mask(
        class_index, positive_only=False, negative_only=True,
        num_features=10, hide_rest=True,
    )
    axes[2].imshow(mark_boundaries(temp_neg, mask_neg, color=(1, 0, 0)))
    axes[2].set_title(f"{CLASS_NAMES[class_index]} — Negative", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"{model_name} — LIME Positive/Negative", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def plot_gradcam_vs_lime(image, gradcam_overlay, lime_explanation, class_index,
                         model_name, save_path=None):
    """Side-by-side comparison of Grad-CAM and LIME for the same image."""
    if image.max() > 1.0:
        image = image / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Grad-CAM
    axes[1].imshow(gradcam_overlay)
    axes[1].set_title(f"Grad-CAM ({CLASS_NAMES[class_index]})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # LIME
    temp, mask = lime_explanation.get_image_and_mask(
        class_index, positive_only=False, num_features=10, hide_rest=False,
    )
    bounded = mark_boundaries(temp, mask, color=(1, 0, 0), mode="thick")
    axes[2].imshow(bounded)
    axes[2].set_title(f"LIME ({CLASS_NAMES[class_index]})", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(f"{model_name} — Grad-CAM vs LIME", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def generate_lime_for_samples(model, dataset, model_name, n_per_class=3, save_dir=None):
    """Generate LIME explanations for sample images from each class."""
    if save_dir is None:
        save_dir = os.path.join(LIME_DIR, model_name)
    os.makedirs(save_dir, exist_ok=True)

    explainer = create_lime_explainer()
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
            explanation = explain_image(model, img, explainer=explainer, num_samples=500)

            # Full LIME plot
            save_path = os.path.join(save_dir, f"{class_name}_sample{i+1}_lime.png")
            plot_lime_explanation(img, explanation, model_name,
                                 predicted_class=class_idx, save_path=save_path)

            # Positive/negative plot
            save_path = os.path.join(save_dir, f"{class_name}_sample{i+1}_posneg.png")
            plot_lime_positive_negative(img, explanation, class_idx, model_name,
                                        save_path=save_path)

    print(f"LIME explanations saved to: {save_dir}")
