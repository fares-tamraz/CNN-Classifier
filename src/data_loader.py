"""Dataset loading utilities.

Expects a folder-based classification dataset:
  <data_dir>/train/<class_name>/*.jpg
  <data_dir>/val/<class_name>/*.jpg
  <data_dir>/test/<class_name>/*.jpg

Use tools/prepare_dataset.py to generate this from a Roboflow YOLO export.
"""

import os
from pathlib import Path
from typing import Tuple, List, Optional

import tensorflow as tf


def build_augmentation_layer() -> tf.keras.Sequential:
    """Mild augmentations that generally help with conveyor/camera variation."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomTranslation(0.03, 0.03),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="augment",
    )


def load_datasets(
    data_dir: str,
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    color_mode: str = "grayscale",
    augment: bool = False,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load train/val/test datasets.

    Args:
        data_dir: path to processed classification dataset root.
        image_size: (H, W)
        batch_size: batch size
        color_mode: "grayscale" or "rgb"
        augment: whether to apply augmentation to training set
        seed: random seed for shuffling

    Returns:
        (train_ds, val_ds, test_ds, class_names)
    """

    data_dir = str(Path(data_dir).resolve())

    def _load(split: str, shuffle: bool) -> tf.data.Dataset:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"Missing split folder: {split_dir}")

        return tf.keras.utils.image_dataset_from_directory(
            split_dir,
            labels="inferred",
            label_mode="int",  # works with sparse_categorical_crossentropy
            color_mode=color_mode,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )

    train_ds = _load("train", shuffle=True)
    val_ds = _load("val", shuffle=True)
    test_ds = _load("test", shuffle=False)

    class_names = list(train_ds.class_names)

    # Normalize to [0,1]
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)

    if augment:
        aug = build_augmentation_layer()
        train_ds = train_ds.map(lambda x, y: (aug(rescale(x), training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    else:
        train_ds = train_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
