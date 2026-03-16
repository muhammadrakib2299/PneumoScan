"""
PneumoScan — Centralized Configuration
All hyperparameters, paths, and constants in one place.
"""

import os

# =============================================================================
# PATHS
# =============================================================================

# Base project directory (works in both local and Colab)
# In Colab, override BASE_DIR to your Google Drive mount path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw", "chest_xray")
TRAIN_DIR = os.path.join(RAW_DATA_DIR, "train")
VAL_DIR = os.path.join(RAW_DATA_DIR, "val")
TEST_DIR = os.path.join(RAW_DATA_DIR, "test")
SAMPLE_DIR = os.path.join(DATA_DIR, "sample")

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")

# Output paths
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
TFLITE_DIR = os.path.join(OUTPUTS_DIR, "tflite")

# Figure subdirectories
EDA_FIGURES_DIR = os.path.join(FIGURES_DIR, "eda")
TRAINING_CURVES_DIR = os.path.join(FIGURES_DIR, "training_curves")
CONFUSION_MATRICES_DIR = os.path.join(FIGURES_DIR, "confusion_matrices")
ROC_CURVES_DIR = os.path.join(FIGURES_DIR, "roc_curves")
PR_CURVES_DIR = os.path.join(FIGURES_DIR, "pr_curves")
GRADCAM_DIR = os.path.join(FIGURES_DIR, "gradcam")
LIME_DIR = os.path.join(FIGURES_DIR, "lime")
COMPARISON_DIR = os.path.join(FIGURES_DIR, "comparison")

# =============================================================================
# CLASS LABELS
# =============================================================================

CLASS_NAMES = ["NORMAL", "BACTERIA", "VIRUS"]
NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# IMAGE SETTINGS
# =============================================================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
IMG_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 3)

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 32
SEED = 42

# Phase 1: Feature extraction (frozen base)
PHASE1_EPOCHS = 10
PHASE1_LEARNING_RATE = 1e-3

# Phase 2: Fine-tuning (unfreeze top 30%)
PHASE2_EPOCHS = 20
PHASE2_LEARNING_RATE = 1e-5
UNFREEZE_PERCENT = 0.3  # Unfreeze top 30% of base layers

# Callbacks
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN = 1e-7

# Regularization
DROPOUT_RATE = 0.3

# =============================================================================
# AUGMENTATION SETTINGS
# =============================================================================

ROTATION_RANGE = 0.04       # ~15 degrees (in fraction of 2*pi for tf.keras)
ZOOM_RANGE = 0.1            # ±10%
BRIGHTNESS_RANGE = 0.15     # ±15%
TRANSLATION_RANGE = 0.1     # ±10%
HORIZONTAL_FLIP = True

# =============================================================================
# MODEL NAMES
# =============================================================================

MODEL_NAMES = [
    "custom_cnn",
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "mobilenetv2",
]

MODEL_SAVE_PATHS = {
    name: os.path.join(MODELS_DIR, f"{name}.keras")
    for name in MODEL_NAMES
}

ENSEMBLE_CONFIG_PATH = os.path.join(MODELS_DIR, "ensemble_config.json")

# =============================================================================
# EVALUATION
# =============================================================================

CONFIDENCE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# =============================================================================
# GOOGLE COLAB SETTINGS
# =============================================================================

# Set to True when running in Google Colab
USE_COLAB = False

# Google Drive mount path (update if different)
COLAB_DRIVE_DIR = "/content/drive/MyDrive/PneumoScan"
COLAB_CHECKPOINT_DIR = os.path.join(COLAB_DRIVE_DIR, "checkpoints")


def setup_colab():
    """Call this at the top of Colab notebooks to mount Drive and set paths."""
    global BASE_DIR, USE_COLAB
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        USE_COLAB = True
        os.makedirs(COLAB_CHECKPOINT_DIR, exist_ok=True)
        print(f"Colab detected. Checkpoints will save to: {COLAB_CHECKPOINT_DIR}")
    except ImportError:
        print("Not running in Colab. Using local paths.")


def ensure_dirs():
    """Create all output directories if they don't exist."""
    dirs = [
        MODELS_DIR, EDA_FIGURES_DIR, TRAINING_CURVES_DIR,
        CONFUSION_MATRICES_DIR, ROC_CURVES_DIR, PR_CURVES_DIR,
        GRADCAM_DIR, LIME_DIR, COMPARISON_DIR, REPORTS_DIR,
        TFLITE_DIR,
        os.path.join(REPORTS_DIR, "classification_reports"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
