"""
PneumoScan — Model Definitions
All 5 model architectures: Custom CNN, ResNet-50, EfficientNet-B0, DenseNet-121, MobileNetV2.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import IMG_SHAPE, NUM_CLASSES, DROPOUT_RATE


def _build_classification_head(base_model, freeze=True):
    """
    Attach a classification head to a pretrained base model.

    Architecture:
        GlobalAveragePooling2D → Dense(256) + BN + Dropout
        → Dense(128) + BN + Dropout → Dense(NUM_CLASSES, softmax)
    """
    if freeze:
        base_model.trainable = False

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model


def build_custom_cnn():
    """
    Simple 4-block CNN baseline.
    ~500K parameters — serves as the control group.
    """
    model = keras.Sequential([
        layers.Input(shape=IMG_SHAPE),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="custom_cnn")

    return model


def build_resnet50(freeze=True):
    """ResNet-50 with ImageNet weights + custom classification head."""
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE,
    )
    model = _build_classification_head(base, freeze=freeze)
    model._name = "resnet50"
    return model


def build_efficientnet_b0(freeze=True):
    """EfficientNet-B0 with ImageNet weights + custom classification head."""
    base = keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE,
    )
    model = _build_classification_head(base, freeze=freeze)
    model._name = "efficientnet_b0"
    return model


def build_densenet121(freeze=True):
    """DenseNet-121 with ImageNet weights + custom classification head (CheXNet architecture)."""
    base = keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE,
    )
    model = _build_classification_head(base, freeze=freeze)
    model._name = "densenet121"
    return model


def build_mobilenetv2(freeze=True):
    """MobileNetV2 with ImageNet weights + custom classification head (lightweight)."""
    base = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=IMG_SHAPE,
    )
    model = _build_classification_head(base, freeze=freeze)
    model._name = "mobilenetv2"
    return model


def unfreeze_top_layers(model, percent=0.3):
    """
    Unfreeze the top `percent` of layers in the base model for fine-tuning.

    Args:
        model: Keras model with a pretrained base.
        percent: Fraction of layers to unfreeze (from the top).
    """
    # Find the base model (first layer that is a Model)
    base = None
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            base = layer
            break

    if base is None:
        print("No base model found. Making all layers trainable.")
        model.trainable = True
        return model

    total_layers = len(base.layers)
    freeze_until = int(total_layers * (1 - percent))

    base.trainable = True
    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    trainable = sum(1 for layer in base.layers if layer.trainable)
    print(f"Unfreezing: {trainable}/{total_layers} base layers ({percent*100:.0f}%)")
    return model


MODEL_BUILDERS = {
    "custom_cnn": build_custom_cnn,
    "resnet50": build_resnet50,
    "efficientnet_b0": build_efficientnet_b0,
    "densenet121": build_densenet121,
    "mobilenetv2": build_mobilenetv2,
}


def get_model(name, freeze=True):
    """Get a model by name."""
    if name == "custom_cnn":
        return build_custom_cnn()
    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_BUILDERS.keys())}")
    return builder(freeze=freeze)
