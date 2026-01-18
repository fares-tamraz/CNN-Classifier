"""Dataset loading utilities.

Folder format:
  <data_dir>/train/<class_name>/*
  <data_dir>/val/<class_name>/*
  <data_dir>/test/<class_name>/*

Keras infers class indices from folder names (alphabetical).
This module enforces that *all splits* use the same inferred order.

Use tools/prepare_dataset.py to generate this dataset from a Roboflow YOLO export.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

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
    shuffle_val: bool = False,
    verify_class_order: bool = True,
    verbose: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    """Load train/val/test datasets.

    Args:
        data_dir: path to processed classification dataset root.
        image_size: (H, W).
        batch_size: batch size.
        color_mode: "grayscale" or "rgb".
        augment: apply augmentation to training set.
        seed: random seed for shuffling.
        shuffle_val: shuffle the validation split (usually False).
        verify_class_order: assert class order is identical across splits.
        verbose: print split summaries.

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
    val_ds = _load("val", shuffle=shuffle_val)
    test_ds = _load("test", shuffle=False)

    class_names = list(train_ds.class_names)

    if verbose:
        print("TRAIN classes:", train_ds.class_names)
        print("VAL classes:  ", val_ds.class_names)
        print("TEST classes: ", test_ds.class_names)

    if verify_class_order:
        if list(val_ds.class_names) != class_names:
            raise ValueError(
                f"Val class order mismatch! train={class_names} val={list(val_ds.class_names)}"
            )
        if list(test_ds.class_names) != class_names:
            raise ValueError(
                f"Test class order mismatch! train={class_names} test={list(test_ds.class_names)}"
            )

    # Normalize to [0,1]
    rescale = tf.keras.layers.Rescaling(1.0 / 255.0)

    if augment:
        aug = build_augmentation_layer()
        train_ds = train_ds.map(
            lambda x, y: (aug(rescale(x), training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        train_ds = train_ds.map(
            lambda x, y: (rescale(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    val_ds = val_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (rescale(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    # Performance
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
