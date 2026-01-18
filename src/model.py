"""Model definitions.

build_model() returns an *uncompiled* Keras model.
Compilation happens in train.py so loss/metrics stay consistent.
"""

import tensorflow as tf


def build_model(
    input_shape=(224, 224, 1),
    num_classes=2,
    model_type="simple",
    dropout=0.3,
):
    """Build a CNN for grayscale X-ray style cap inspection.

    Args:
        input_shape: (H, W, C). Use C=1 for grayscale.
        num_classes: 2 for binary, 5 for multi-class, etc.
        model_type: "simple" or "mobilenetv2".
        dropout: dropout rate on the classifier head.

    Returns:
        tf.keras.Model (uncompiled)
    """

    if model_type not in {"simple", "mobilenetv2"}:
        raise ValueError("model_type must be 'simple' or 'mobilenetv2'")

    if model_type == "simple":
        return _build_simple_cnn(input_shape=input_shape, num_classes=num_classes, dropout=dropout)

    return _build_mobilenetv2(input_shape=input_shape, num_classes=num_classes, dropout=dropout)


def _build_simple_cnn(input_shape, num_classes, dropout):
    inputs = tf.keras.Input(shape=input_shape)

    x = inputs

    # Block 1
    x = tf.keras.layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 2
    x = tf.keras.layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 3
    x = tf.keras.layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)

    # Block 4
    x = tf.keras.layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cap_cnn_simple")


def _build_mobilenetv2(input_shape, num_classes, dropout):
    """Transfer learning option.

    Notes:
      - MobileNetV2 expects 3 channels. If you pass grayscale (C=1), we
        convert to RGB by repeating channels.
      - This is optional; keep 'simple' as your baseline.
    """

    inputs = tf.keras.Input(shape=input_shape)

    if input_shape[-1] == 1:
        x = tf.keras.layers.Lambda(lambda t: tf.image.grayscale_to_rgb(t))(inputs)
    else:
        x = inputs

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(input_shape[0], input_shape[1], 3),
    )
    base.trainable = False

    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="cap_cnn_mobilenetv2")
