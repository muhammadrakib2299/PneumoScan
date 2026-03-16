"""
PneumoScan — Ensemble Model
Combines top-3 models using soft voting or weighted voting.
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import (
    MODEL_NAMES, MODEL_SAVE_PATHS, ENSEMBLE_CONFIG_PATH,
    MODELS_DIR, CLASS_NAMES,
)


def load_models(model_names=None):
    """Load all saved models."""
    if model_names is None:
        model_names = MODEL_NAMES

    models = {}
    for name in model_names:
        path = MODEL_SAVE_PATHS.get(name)
        if path and os.path.exists(path):
            print(f"Loading: {name} from {path}")
            models[name] = keras.models.load_model(path)
        else:
            print(f"Skipping {name} — not found at {path}")
    return models


def rank_models_by_metric(models, val_ds, metric="accuracy"):
    """
    Rank models by a given metric on the validation set.

    Returns:
        sorted list of (model_name, metric_value) tuples
    """
    from evaluate import predict_dataset
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    rankings = []
    for name, model in models.items():
        y_true_oh, y_true, y_pred_proba, y_pred = predict_dataset(model, val_ds)

        if metric == "accuracy":
            score = accuracy_score(y_true, y_pred)
        elif metric == "f1_macro":
            score = f1_score(y_true, y_pred, average="macro")
        elif metric == "auc_roc":
            try:
                score = roc_auc_score(y_true_oh, y_pred_proba, multi_class="ovr", average="macro")
            except ValueError:
                score = 0.0
        else:
            score = accuracy_score(y_true, y_pred)

        rankings.append((name, score))
        print(f"  {name}: {metric} = {score:.4f}")

    rankings.sort(key=lambda x: x[1], reverse=True)
    return rankings


def soft_voting_predict(models_dict, dataset):
    """
    Ensemble prediction using soft voting (average probabilities).

    Args:
        models_dict: {model_name: model}
        dataset: tf.data.Dataset

    Returns:
        y_pred_proba: averaged probability predictions
    """
    all_preds = []
    for name, model in models_dict.items():
        preds = []
        for images, _ in dataset:
            batch_preds = model.predict(images, verbose=0)
            preds.append(batch_preds)
        all_preds.append(np.concatenate(preds, axis=0))

    # Average probabilities
    avg_preds = np.mean(all_preds, axis=0)
    return avg_preds


def weighted_voting_predict(models_dict, weights, dataset):
    """
    Ensemble prediction using weighted voting.

    Args:
        models_dict: {model_name: model}
        weights: {model_name: weight}
        dataset: tf.data.Dataset

    Returns:
        y_pred_proba: weighted probability predictions
    """
    all_preds = []
    model_weights = []

    for name, model in models_dict.items():
        preds = []
        for images, _ in dataset:
            batch_preds = model.predict(images, verbose=0)
            preds.append(batch_preds)
        all_preds.append(np.concatenate(preds, axis=0))
        model_weights.append(weights.get(name, 1.0))

    # Normalize weights
    total_weight = sum(model_weights)
    model_weights = [w / total_weight for w in model_weights]

    # Weighted average
    weighted_preds = np.zeros_like(all_preds[0])
    for preds, weight in zip(all_preds, model_weights):
        weighted_preds += weight * preds

    return weighted_preds


def build_ensemble(models_dict, val_ds, top_k=3):
    """
    Build ensemble from top-K models.

    Returns:
        top_models: dict of top-K models
        weights: dict of model weights (based on AUC scores)
        config: ensemble configuration dict
    """
    print("\nRanking models by AUC-ROC...")
    rankings = rank_models_by_metric(models_dict, val_ds, metric="auc_roc")

    # Select top-K
    top_names = [name for name, _ in rankings[:top_k]]
    top_models = {name: models_dict[name] for name in top_names}

    # Weights based on AUC scores
    weights = {name: score for name, score in rankings[:top_k]}

    print(f"\nTop-{top_k} models for ensemble:")
    for name, score in rankings[:top_k]:
        print(f"  {name}: AUC-ROC = {score:.4f}")

    # Save config
    config = {
        "top_k": top_k,
        "models": top_names,
        "weights": weights,
        "ranking": {name: float(score) for name, score in rankings},
    }

    os.makedirs(os.path.dirname(ENSEMBLE_CONFIG_PATH), exist_ok=True)
    with open(ENSEMBLE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Ensemble config saved to: {ENSEMBLE_CONFIG_PATH}")

    return top_models, weights, config


def evaluate_ensemble(top_models, weights, test_ds):
    """Evaluate both soft voting and weighted voting ensembles."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Get true labels
    y_true = []
    for _, labels in test_ds:
        y_true.append(labels.numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_true_labels = np.argmax(y_true, axis=1)

    # Soft voting
    print("\n--- Soft Voting Ensemble ---")
    soft_preds = soft_voting_predict(top_models, test_ds)
    soft_labels = np.argmax(soft_preds, axis=1)
    soft_acc = accuracy_score(y_true_labels, soft_labels)
    soft_f1 = f1_score(y_true_labels, soft_labels, average="macro")
    try:
        soft_auc = roc_auc_score(y_true, soft_preds, multi_class="ovr", average="macro")
    except ValueError:
        soft_auc = 0.0
    print(f"  Accuracy: {soft_acc:.4f}")
    print(f"  F1 Macro: {soft_f1:.4f}")
    print(f"  AUC-ROC:  {soft_auc:.4f}")

    # Weighted voting
    print("\n--- Weighted Voting Ensemble ---")
    weighted_preds = weighted_voting_predict(top_models, weights, test_ds)
    weighted_labels = np.argmax(weighted_preds, axis=1)
    weighted_acc = accuracy_score(y_true_labels, weighted_labels)
    weighted_f1 = f1_score(y_true_labels, weighted_labels, average="macro")
    try:
        weighted_auc = roc_auc_score(y_true, weighted_preds, multi_class="ovr", average="macro")
    except ValueError:
        weighted_auc = 0.0
    print(f"  Accuracy: {weighted_acc:.4f}")
    print(f"  F1 Macro: {weighted_f1:.4f}")
    print(f"  AUC-ROC:  {weighted_auc:.4f}")

    return {
        "soft_voting": {"accuracy": soft_acc, "f1_macro": soft_f1, "auc_roc": soft_auc, "predictions": soft_preds},
        "weighted_voting": {"accuracy": weighted_acc, "f1_macro": weighted_f1, "auc_roc": weighted_auc, "predictions": weighted_preds},
    }
