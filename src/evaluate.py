"""
PneumoScan — Evaluation Pipeline
Comprehensive metrics: accuracy, F1, AUC-ROC, AUC-PR, confusion matrix, Cohen's Kappa.
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, roc_auc_score, average_precision_score,
)
from config import (
    CLASS_NAMES, NUM_CLASSES, MODEL_NAMES, MODEL_SAVE_PATHS,
    CONFUSION_MATRICES_DIR, ROC_CURVES_DIR, PR_CURVES_DIR,
    COMPARISON_DIR, REPORTS_DIR, CONFIDENCE_THRESHOLDS,
)
from utils import (
    plot_confusion_matrix, plot_confusion_matrix_normalized,
    plot_roc_curves, plot_precision_recall_curves,
    print_classification_report, plot_model_comparison, save_figure,
)


def predict_dataset(model, dataset):
    """Run inference on a dataset and return true labels and predictions."""
    y_true = []
    y_pred_proba = []

    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true.append(labels.numpy())
        y_pred_proba.append(preds)

    y_true = np.concatenate(y_true, axis=0)
    y_pred_proba = np.concatenate(y_pred_proba, axis=0)
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred_proba, axis=1)

    return y_true, y_true_labels, y_pred_proba, y_pred_labels


def measure_inference_time(model, dataset, n_batches=5):
    """Measure average inference time per image in milliseconds."""
    times = []
    for i, (images, _) in enumerate(dataset):
        if i >= n_batches:
            break
        start = time.time()
        model.predict(images, verbose=0)
        elapsed = time.time() - start
        times.append(elapsed / images.shape[0])

    avg_ms = np.mean(times) * 1000
    return avg_ms


def evaluate_model(model, test_ds, model_name):
    """
    Full evaluation pipeline for a single model.

    Returns:
        dict with all metrics
    """
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 60}")

    # Get predictions
    y_true_onehot, y_true, y_pred_proba, y_pred = predict_dataset(model, test_ds)

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)

    # AUC scores
    try:
        auc_roc = roc_auc_score(y_true_onehot, y_pred_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc_roc = 0.0

    try:
        auc_pr = average_precision_score(y_true_onehot, y_pred_proba, average="macro")
    except ValueError:
        auc_pr = 0.0

    # Inference time
    inference_ms = measure_inference_time(model, test_ds)

    # Print results
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  F1 (macro):   {f1_macro:.4f}")
    print(f"  F1 (weighted):{f1_weighted:.4f}")
    print(f"  AUC-ROC:      {auc_roc:.4f}")
    print(f"  AUC-PR:       {auc_pr:.4f}")
    print(f"  Cohen's Kappa:{kappa:.4f}")
    print(f"  Inference:    {inference_ms:.2f} ms/image")

    # Classification report
    report = print_classification_report(y_true, y_pred, model_name)

    # Save classification report
    report_path = os.path.join(REPORTS_DIR, "classification_reports", f"{model_name}_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)

    # Plot confusion matrices
    plot_confusion_matrix(
        y_true, y_pred, model_name,
        save_path=os.path.join(CONFUSION_MATRICES_DIR, f"{model_name}_cm.png"),
    )
    plot_confusion_matrix_normalized(
        y_true, y_pred, model_name,
        save_path=os.path.join(CONFUSION_MATRICES_DIR, f"{model_name}_cm_normalized.png"),
    )

    # Plot ROC curves
    _, auc_scores = plot_roc_curves(
        y_true_onehot, y_pred_proba, model_name,
        save_path=os.path.join(ROC_CURVES_DIR, f"{model_name}_roc.png"),
    )

    # Plot PR curves
    _, ap_scores = plot_precision_recall_curves(
        y_true_onehot, y_pred_proba, model_name,
        save_path=os.path.join(PR_CURVES_DIR, f"{model_name}_pr.png"),
    )

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "cohens_kappa": kappa,
        "inference_ms": inference_ms,
        "auc_per_class": auc_scores,
        "ap_per_class": ap_scores,
    }

    return metrics


def multi_threshold_analysis(model, test_ds, model_name, save_path=None):
    """Analyze precision/recall/F1 at different confidence thresholds."""
    import matplotlib.pyplot as plt

    y_true_onehot, y_true, y_pred_proba, _ = predict_dataset(model, test_ds)

    results = []
    for threshold in CONFIDENCE_THRESHOLDS:
        # Apply threshold — predict class only if max prob >= threshold
        max_probs = np.max(y_pred_proba, axis=1)
        confident_mask = max_probs >= threshold
        y_pred_thresh = np.argmax(y_pred_proba[confident_mask], axis=1)
        y_true_thresh = y_true[confident_mask]

        if len(y_true_thresh) == 0:
            continue

        acc = accuracy_score(y_true_thresh, y_pred_thresh)
        f1 = f1_score(y_true_thresh, y_pred_thresh, average="macro", zero_division=0)
        coverage = confident_mask.sum() / len(y_true)

        results.append({
            "threshold": threshold,
            "accuracy": acc,
            "f1_macro": f1,
            "coverage": coverage,
        })

    df = pd.DataFrame(results)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(df["threshold"], df["accuracy"], "b-o", linewidth=2, label="Accuracy")
    ax1.plot(df["threshold"], df["f1_macro"], "r-s", linewidth=2, label="F1 (Macro)")
    ax2.plot(df["threshold"], df["coverage"], "g--^", linewidth=2, label="Coverage", alpha=0.7)

    ax1.set_xlabel("Confidence Threshold", fontsize=12)
    ax1.set_ylabel("Score", fontsize=12, color="blue")
    ax2.set_ylabel("Coverage (% of predictions)", fontsize=12, color="green")
    ax1.set_title(f"{model_name} — Multi-Threshold Analysis", fontsize=14, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left")
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        from utils import save_figure
        save_figure(fig, save_path)
    else:
        plt.show()

    return df


def evaluate_all_models(test_ds, model_names=None):
    """Evaluate all saved models and generate comparison."""
    if model_names is None:
        model_names = MODEL_NAMES

    all_metrics = {}

    for name in model_names:
        model_path = MODEL_SAVE_PATHS.get(name)
        if model_path and os.path.exists(model_path):
            print(f"\nLoading: {model_path}")
            model = keras.models.load_model(model_path)
            metrics = evaluate_model(model, test_ds, name)
            all_metrics[name] = metrics
        else:
            print(f"Skipping {name} — model file not found at {model_path}")

    # Generate comparison table
    if all_metrics:
        comparison_df = pd.DataFrame([
            {
                "Model": m["model_name"],
                "Accuracy": f"{m['accuracy']:.4f}",
                "F1 (Macro)": f"{m['f1_macro']:.4f}",
                "AUC-ROC": f"{m['auc_roc']:.4f}",
                "AUC-PR": f"{m['auc_pr']:.4f}",
                "Cohen's Kappa": f"{m['cohens_kappa']:.4f}",
                "Inference (ms)": f"{m['inference_ms']:.2f}",
            }
            for m in all_metrics.values()
        ])

        csv_path = os.path.join(REPORTS_DIR, "model_comparison.csv")
        comparison_df.to_csv(csv_path, index=False)
        print(f"\nComparison table saved to: {csv_path}")
        print(comparison_df.to_string(index=False))

        # Plot comparison charts
        results_for_plot = {
            m["model_name"]: {"accuracy": m["accuracy"], "f1_macro": m["f1_macro"], "auc_roc": m["auc_roc"]}
            for m in all_metrics.values()
        }
        for metric in ["accuracy", "f1_macro", "auc_roc"]:
            plot_model_comparison(
                results_for_plot, metric=metric,
                save_path=os.path.join(COMPARISON_DIR, f"comparison_{metric}.png"),
            )

    return all_metrics


if __name__ == "__main__":
    from data_loader import load_test_dataset
    import argparse

    parser = argparse.ArgumentParser(description="PneumoScan Evaluation")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    args = parser.parse_args()

    test_ds = load_test_dataset()

    if args.model == "all":
        evaluate_all_models(test_ds)
    else:
        model_path = MODEL_SAVE_PATHS.get(args.model)
        model = keras.models.load_model(model_path)
        evaluate_model(model, test_ds, args.model)
