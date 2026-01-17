import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn_model(
    input_shape=(224, 224, 3),
    num_classes=2
):
    """
    Builds and returns a CNN model for image classification.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes

    Returns:
        model (tf.keras.Model): Compiled CNN model
    """

    model = models.Sequential()

    # -----------------------------
    # Convolutional Block 1
    # -----------------------------
    model.add(layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=input_shape
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # -----------------------------
    # Convolutional Block 2
    # -----------------------------
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # -----------------------------
    # Convolutional Block 3
    # -----------------------------
    model.add(layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        activation='relu'
    ))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # -----------------------------
    # Fully Connected Layers
    # -----------------------------
    model.add(layers.Flatten())

    model.add(layers.Dense(
        units=128,
        activation='relu'
    ))

    model.add(layers.Dropout(0.5))

    # Output layer
    model.add(layers.Dense(
        units=num_classes,
        activation='softmax'
    ))

    # -----------------------------
    # Compile Model
    # -----------------------------
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
