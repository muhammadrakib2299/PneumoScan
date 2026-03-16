"""
PneumoScan — Training Pipeline
Two-phase training: feature extraction (frozen) → fine-tuning (unfrozen top 30%).
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from config import (
    PHASE1_EPOCHS, PHASE1_LEARNING_RATE,
    PHASE2_EPOCHS, PHASE2_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE,
    REDUCE_LR_FACTOR, REDUCE_LR_MIN,
    UNFREEZE_PERCENT, MODELS_DIR, TRAINING_CURVES_DIR,
    MODEL_SAVE_PATHS, USE_COLAB, COLAB_CHECKPOINT_DIR,
)
from models import get_model, unfreeze_top_layers
from utils import plot_training_history


def get_callbacks(model_name, phase="phase1"):
    """Build training callbacks."""
    # Checkpoint path
    if USE_COLAB:
        ckpt_dir = COLAB_CHECKPOINT_DIR
    else:
        ckpt_dir = MODELS_DIR

    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{model_name}_{phase}_best.keras")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=REDUCE_LR_MIN,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]
    return callbacks


def compile_model(model, learning_rate):
    """Compile model with Adam optimizer and categorical crossentropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(
    model_name,
    train_ds,
    val_ds,
    class_weights=None,
    phase1_epochs=None,
    phase2_epochs=None,
):
    """
    Full two-phase training pipeline.

    Phase 1: Feature extraction (frozen base)
    Phase 2: Fine-tuning (unfreeze top 30%)

    Returns:
        model, history_phase1, history_phase2
    """
    if phase1_epochs is None:
        phase1_epochs = PHASE1_EPOCHS
    if phase2_epochs is None:
        phase2_epochs = PHASE2_EPOCHS

    print(f"\n{'=' * 60}")
    print(f"Training: {model_name}")
    print(f"{'=' * 60}")

    # Build model
    is_custom = model_name == "custom_cnn"
    model = get_model(model_name, freeze=not is_custom)

    # ── Phase 1: Feature extraction ──
    print(f"\n--- Phase 1: {'Full training' if is_custom else 'Feature Extraction (frozen base)'} ---")
    model = compile_model(model, PHASE1_LEARNING_RATE)
    model.summary(show_trainable=True, expand_nested=False)

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=phase1_epochs,
        class_weight=class_weights,
        callbacks=get_callbacks(model_name, "phase1"),
    )

    # Save phase 1 training curves
    save_path = os.path.join(TRAINING_CURVES_DIR, f"{model_name}_phase1.png")
    plot_training_history(history1, f"{model_name} (Phase 1)", save_path=save_path)

    # ── Phase 2: Fine-tuning (skip for custom CNN) ──
    history2 = None
    if not is_custom:
        print(f"\n--- Phase 2: Fine-Tuning (unfreeze top {UNFREEZE_PERCENT*100:.0f}%) ---")
        model = unfreeze_top_layers(model, percent=UNFREEZE_PERCENT)
        model = compile_model(model, PHASE2_LEARNING_RATE)

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=phase2_epochs,
            class_weight=class_weights,
            callbacks=get_callbacks(model_name, "phase2"),
        )

        save_path = os.path.join(TRAINING_CURVES_DIR, f"{model_name}_phase2.png")
        plot_training_history(history2, f"{model_name} (Phase 2)", save_path=save_path)

    # ── Save final model ──
    save_path = MODEL_SAVE_PATHS.get(model_name, os.path.join(MODELS_DIR, f"{model_name}.keras"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    return model, history1, history2


def train_all_models(train_ds, val_ds, class_weights=None, model_names=None):
    """Train all models sequentially."""
    from config import MODEL_NAMES

    if model_names is None:
        model_names = MODEL_NAMES

    results = {}
    for name in model_names:
        model, h1, h2 = train_model(name, train_ds, val_ds, class_weights)
        results[name] = {"model": model, "history_phase1": h1, "history_phase2": h2}

    return results


if __name__ == "__main__":
    from data_loader import load_train_val_split, compute_class_weights
    from config import TRAIN_DIR

    train_ds, val_ds = load_train_val_split()
    class_weights = compute_class_weights()

    import argparse
    parser = argparse.ArgumentParser(description="PneumoScan Training")
    parser.add_argument("--model", type=str, default="all", help="Model name or 'all'")
    args = parser.parse_args()

    if args.model == "all":
        train_all_models(train_ds, val_ds, class_weights)
    else:
        train_model(args.model, train_ds, val_ds, class_weights)
