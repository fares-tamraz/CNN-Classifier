#!/usr/bin/env python3
"""
Train v7 model on 3-class dataset (cleaned labels).

Changes from v6:
1. Dataset cleaned — 44 mislabeled images removed after manual review
2. tf.random.set_seed() added for reproducibility
3. Rich terminal progress: per-epoch timing + total time remaining
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_loader import load_datasets
from src.losses import FocalLoss


# ============================================================================
# REPRODUCIBILITY
# ============================================================================
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# PROGRESS CALLBACK
# ============================================================================
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    """
    Prints a clear per-epoch summary with:
      - Epoch N / total
      - Train accuracy & loss
      - Val accuracy & loss  (with delta vs best)
      - Current learning rate
      - Epoch duration
      - Estimated time remaining for full training
    """

    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0
        self._epoch_times: list = []
        self._best_val_acc: float = 0.0

    def on_train_begin(self, logs=None):
        self._train_start = time.time()
        print("\n" + "=" * 70)
        print(f"  TRAINING START  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70 + "\n")

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        elapsed_epoch = time.time() - self._epoch_start
        self._epoch_times.append(elapsed_epoch)

        current_epoch = epoch + 1
        train_acc = logs.get("accuracy", 0.0)
        val_acc = logs.get("val_accuracy", 0.0)
        train_loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)

        # Learning rate
        try:
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        except Exception:
            lr = float("nan")

        # Val accuracy delta vs best
        delta = val_acc - self._best_val_acc
        if val_acc > self._best_val_acc:
            self._best_val_acc = val_acc
            best_marker = " *** NEW BEST ***"
        else:
            best_marker = ""

        # Time estimates
        avg_epoch_time = sum(self._epoch_times) / len(self._epoch_times)
        epochs_remaining = self.total_epochs - current_epoch
        eta_seconds = avg_epoch_time * epochs_remaining
        eta_str = str(timedelta(seconds=int(eta_seconds)))
        total_elapsed = str(timedelta(seconds=int(time.time() - self._train_start)))

        # Delta string
        delta_str = f"{delta:+.4f}" if delta != 0 else " 0.0000"

        print("\n" + "-" * 70)
        print(
            f"  Epoch {current_epoch:>3}/{self.total_epochs}  |  "
            f"elapsed: {total_elapsed}  |  ETA: {eta_str}"
        )
        print(
            f"  Train  —  acc: {train_acc:.4f}   loss: {train_loss:.4f}"
        )
        print(
            f"  Val    —  acc: {val_acc:.4f}   loss: {val_loss:.4f}   "
            f"Δbest: {delta_str}{best_marker}"
        )
        print(
            f"  Best val acc so far: {self._best_val_acc:.4f}   "
            f"lr: {lr:.2e}   epoch took: {elapsed_epoch:.1f}s"
        )
        print("-" * 70)

    def on_train_end(self, logs=None):
        total = str(timedelta(seconds=int(time.time() - self._train_start)))
        print("\n" + "=" * 70)
        print(f"  TRAINING FINISHED  —  total time: {total}")
        print(f"  Best val accuracy: {self._best_val_acc:.4f}")
        print("=" * 70 + "\n")


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
def build_simple_cnn(input_shape, num_classes, dropout=0.4):
    """Build CNN — augmentation handled in data pipeline, not model."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Block 1: 32 filters
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # Block 2: 64 filters
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout * 0.5)(x)

    # Block 3: 128 filters
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Dropout(dropout * 0.75)(x)

    # Block 4: 256 filters
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Conv2D(256, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    # Dense layers
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout * 0.75)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# ============================================================================
# CLASS WEIGHTS
# ============================================================================
def compute_class_weights(y_train: np.ndarray, boost_minority: float = 1.5) -> dict:
    """Balanced class weights with optional minority boosting."""
    classes, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(classes)

    weights = {}
    for cls, count in zip(classes, counts):
        weight = n_samples / (n_classes * count)
        weights[int(cls)] = weight

    max_weight = max(weights.values())
    for cls in weights:
        if weights[cls] > 1.0:
            weights[cls] = min(weights[cls] * boost_minority, max_weight * 2)

    return weights


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("V7 3-CLASS MODEL TRAINING  (cleaned dataset — mislabels removed)")
    print("=" * 70)

    # ── Configuration ────────────────────────────────────────────────────────
    data_dir = "data/processed/cls_3class_crops"
    out_dir = Path("models/v7")
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = (224, 224)
    batch_size = 32
    epochs = 50
    initial_lr = 1e-4
    dropout = 0.4
    label_smoothing = 0.1
    focal_gamma = 2.5
    patience = 6

    print(f"\nConfiguration:")
    print(f"  Data dir    : {data_dir}")
    print(f"  Output dir  : {out_dir}")
    print(f"  Image size  : {image_size}")
    print(f"  Batch size  : {batch_size}")
    print(f"  Max epochs  : {epochs}")
    print(f"  Initial LR  : {initial_lr}")
    print(f"  Dropout     : {dropout}")
    print(f"  Label smooth: {label_smoothing}")
    print(f"  Focal gamma : {focal_gamma}")
    print(f"  ES patience : {patience}  (LR-reduce patience: 3)")
    print(f"  Seed        : {SEED}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        augment=True,
        seed=SEED,
    )

    num_classes = len(class_names)
    print(f"\nClasses ({num_classes}): {class_names}")

    class_names_path = out_dir / "class_names.json"
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    # Class weights from train labels
    y_train = []
    for _, labels in train_ds:
        y_train.extend(labels.numpy())
    y_train = np.array(y_train)

    class_weights = compute_class_weights(y_train, boost_minority=1.5)
    print(f"\nClass weights:")
    for idx, name in enumerate(class_names):
        print(f"  [{idx}] {name}: {class_weights[idx]:.3f}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    input_shape = (*image_size, 1)  # grayscale
    model = build_simple_cnn(input_shape, num_classes, dropout=dropout)

    loss_fn = FocalLoss(alpha=1.0, gamma=focal_gamma, label_smoothing=label_smoothing)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-5)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_path = out_dir / "cap_classifier_best.keras"
    log_dir = out_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")

    callbacks = [
        # Rich per-epoch summary with ETA
        TrainingProgressCallback(total_epochs=epochs),

        tf.keras.callbacks.ModelCheckpoint(
            best_path.as_posix(),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir.as_posix(),
            histogram_freq=0,
            write_graph=False,
        ),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,  # Keras batch-level progress bar
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    final_path = out_dir / "cap_classifier_final.keras"
    model.save(final_path.as_posix())

    best_val_acc = max(history.history.get("val_accuracy", [0]))
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1

    print(f"Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"Models saved to   : {out_dir}")
    print(f"  Best   : {best_path}")
    print(f"  Final  : {final_path}")
    print(f"  Classes: {class_names_path}")
    print(f"  TB logs: {log_dir}")

    # ── Quick test eval ───────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("TEST SET EVALUATION")
    print("-" * 70)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")
    print(
        "\nDone. Run:  python src/evaluate.py --model models/v7/cap_classifier_best.keras"
    )


if __name__ == "__main__":
    main()
