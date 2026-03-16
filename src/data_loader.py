"""
PneumoScan — Data Loading Pipeline
tf.data pipeline with augmentation, class weights, and stratified splits.
"""

import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import config
from config import (
    IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES, NUM_CLASSES,
    ROTATION_RANGE, ZOOM_RANGE, BRIGHTNESS_RANGE,
    TRANSLATION_RANGE, HORIZONTAL_FLIP,
)


def build_augmentation_layer():
    """Build a Keras augmentation layer for training data."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal" if HORIZONTAL_FLIP else "none"),
        tf.keras.layers.RandomRotation(ROTATION_RANGE),
        tf.keras.layers.RandomZoom(ZOOM_RANGE),
        tf.keras.layers.RandomTranslation(TRANSLATION_RANGE, TRANSLATION_RANGE),
        tf.keras.layers.RandomBrightness(BRIGHTNESS_RANGE),
    ], name="augmentation")


def load_dataset_from_directory(directory, shuffle=True, augment=False):
    """
    Load images from a directory using tf.keras.utils.image_dataset_from_directory.

    Args:
        directory: Path to dataset directory with class subdirectories.
        shuffle: Whether to shuffle the dataset.
        augment: Whether to apply augmentation.

    Returns:
        tf.data.Dataset, class_names list
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=shuffle,
        seed=SEED,
    )

    if augment:
        augmentation = build_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Normalize to [0, 1]
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)
    dataset = dataset.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def load_train_dataset(train_dir=None, augment=True):
    """Load training dataset with augmentation."""
    if train_dir is None:
        train_dir = config.TRAIN_DIR
    return load_dataset_from_directory(train_dir, shuffle=True, augment=augment)


def load_test_dataset(test_dir=None):
    """Load test dataset without augmentation."""
    if test_dir is None:
        test_dir = config.TEST_DIR
    return load_dataset_from_directory(test_dir, shuffle=False, augment=False)


def load_train_val_split(train_dir=None, val_split=0.2, augment=True):
    """
    Load training data with a validation split.

    Returns:
        train_dataset, val_dataset
    """
    if train_dir is None:
        train_dir = config.TRAIN_DIR

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=val_split,
        subset="training",
        seed=SEED,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        validation_split=val_split,
        subset="validation",
        seed=SEED,
    )

    # Normalize
    normalization = tf.keras.layers.Rescaling(1.0 / 255.0)

    if augment:
        augmentation = build_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (augmentation(normalization(x), training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        train_ds = train_ds.map(
            lambda x, y: (normalization(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    val_ds = val_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def compute_class_weights(train_dir=None):
    """
    Compute class weights to handle imbalanced dataset.

    Returns:
        dict mapping class index to weight
    """
    if train_dir is None:
        train_dir = config.TRAIN_DIR

    class_counts = []
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(train_dir, class_name)
        count = len([
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        ])
        class_counts.append(count)

    total = sum(class_counts)
    weights = {}
    for i, count in enumerate(class_counts):
        weights[i] = total / (NUM_CLASSES * count)

    print("Class weights:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {weights[i]:.4f} ({class_counts[i]} images)")

    return weights


def get_labels_from_directory(directory):
    """Extract all labels from a directory for stratified splitting."""
    labels = []
    filepaths = []
    for i, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                filepaths.append(os.path.join(class_dir, fname))
                labels.append(i)
    return np.array(filepaths), np.array(labels)


def get_kfold_splits(train_dir=None, n_splits=5):
    """
    Generate stratified K-Fold split indices.

    Returns:
        List of (train_indices, val_indices) tuples
    """
    if train_dir is None:
        train_dir = config.TRAIN_DIR

    filepaths, labels = get_labels_from_directory(train_dir)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    splits = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(filepaths, labels)):
        splits.append((train_idx, val_idx))
        print(f"Fold {fold + 1}: train={len(train_idx)}, val={len(val_idx)}")

    return splits, filepaths, labels
