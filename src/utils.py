"""
PneumoScan — Utility Functions
Plotting helpers and common utilities.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from config import CLASS_NAMES, NUM_CLASSES


def save_figure(fig, filepath, dpi=150):
    """Save a matplotlib figure, creating directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {filepath}")


def plot_training_history(history, model_name, save_path=None):
    """Plot training and validation loss + accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[0].set_title(f"{model_name} — Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title(f"{model_name} — Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"{model_name} Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def plot_confusion_matrix_normalized(y_true, y_pred, model_name, save_path=None):
    """Plot normalized confusion matrix (percentages)."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm, annot=True, fmt=".2%", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"{model_name} — Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def plot_roc_curves(y_true_onehot, y_pred_proba, model_name, save_path=None):
    """Plot One-vs-Rest ROC curves for multi-class classification."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#F44336", "#4CAF50"]
    auc_scores = {}

    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = roc_auc
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{class_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} — ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig, auc_scores


def plot_precision_recall_curves(y_true_onehot, y_pred_proba, model_name, save_path=None):
    """Plot Precision-Recall curves for multi-class classification."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#2196F3", "#F44336", "#4CAF50"]
    ap_scores = {}

    for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i], y_pred_proba[:, i]
        )
        ap = average_precision_score(y_true_onehot[:, i], y_pred_proba[:, i])
        ap_scores[class_name] = ap
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{class_name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"{model_name} — Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig, ap_scores


def plot_class_distribution(class_counts, title="Class Distribution", save_path=None):
    """Plot bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ["#2196F3", "#F44336", "#4CAF50"]

    bars = ax.bar(classes, counts, color=colors[:len(classes)], edgecolor="black", alpha=0.8)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(count), ha="center", va="bottom", fontweight="bold", fontsize=12)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def plot_sample_images(dataset, n_per_class=4, save_path=None):
    """Display sample images from each class."""
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class, figsize=(4 * n_per_class, 4 * NUM_CLASSES))

    class_images = {i: [] for i in range(NUM_CLASSES)}

    for images, labels in dataset:
        for img, label in zip(images.numpy(), labels.numpy()):
            class_idx = np.argmax(label)
            if len(class_images[class_idx]) < n_per_class:
                class_images[class_idx].append(img)
        if all(len(v) >= n_per_class for v in class_images.values()):
            break

    for i, class_name in enumerate(CLASS_NAMES):
        for j in range(n_per_class):
            ax = axes[i][j] if NUM_CLASSES > 1 else axes[j]
            if j < len(class_images[i]):
                img = class_images[i][j]
                if img.max() > 1.0:
                    img = img / 255.0
                ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_title(class_name, fontsize=14, fontweight="bold")

    fig.suptitle("Sample Images per Class", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig


def print_classification_report(y_true, y_pred, model_name):
    """Print and return sklearn classification report."""
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    print(f"\n{'=' * 50}")
    print(f"{model_name} — Classification Report")
    print(f"{'=' * 50}")
    print(report)
    return report


def plot_model_comparison(results_dict, metric="accuracy", save_path=None):
    """
    Plot grouped bar chart comparing models on a given metric.

    Args:
        results_dict: {model_name: {metric_name: value}}
        metric: Which metric to plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results_dict.keys())
    values = [results_dict[m].get(metric, 0) for m in models]
    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336"]

    bars = ax.bar(models, values, color=colors[:len(models)], edgecolor="black", alpha=0.85)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Model Comparison — {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    else:
        plt.show()
    return fig
