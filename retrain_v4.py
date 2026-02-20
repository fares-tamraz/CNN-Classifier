#!/usr/bin/env python3
"""Retrain model with improved configuration to fix the collapsed model.

Key improvements:
1. Use simple CNN (trainable end-to-end, not frozen MobileNetV2)
2. Stronger focal loss (gamma=3.0)
3. Heavier class weights
4. More aggressive augmentation
5. Lower learning rate with warmup
6. Better monitoring
"""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.data_loader import load_datasets, build_augmentation_layer
from src.model import build_model


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), tf.shape(y_pred)[-1])
        y_true_onehot = tf.cast(y_true_onehot, y_pred.dtype)
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true_onehot * tf.math.log(y_pred)
        ce = tf.reduce_sum(ce, axis=-1)
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        focal_loss = self.alpha * focal_weight * ce
        return tf.reduce_mean(focal_loss)


def compute_class_weights(y: np.ndarray, boost_factor: float = 2.0) -> dict[int, float]:
    """Compute class weights with extra boost for minority classes."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    k = len(classes)
    
    # Standard balanced weights
    weights = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}
    
    # Boost minority classes even more
    max_weight = max(weights.values())
    for c in weights:
        if weights[c] > 1.0:  # minority class
            weights[c] = min(weights[c] * boost_factor, max_weight * 2)
    
    return weights


def main():
    print("="*80)
    print("RETRAINING V4 - FIXING COLLAPSED MODEL")
    print("="*80 + "\n")
    
    # Configuration
    data_dir = "data/processed/cls_5class_crops_v3"
    out_dir = Path("models/v4")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_size = (224, 224)
    batch_size = 32
    epochs = 30
    
    # Load datasets with stronger augmentation
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
        augment=True,  # Using augmentation from data_loader
        seed=42,
        verify_class_order=True,
        verbose=True,
    )
    
    num_classes = len(class_names)
    print(f"\nClasses: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Compute class weights with boost
    y_train = np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(4096)])
    class_weights = compute_class_weights(y_train, boost_factor=2.5)
    
    print("\nClass distribution (train):")
    for i, name in enumerate(class_names):
        count = int((y_train == i).sum())
        weight = class_weights[i]
        print(f"  {name:15s}: {count:4d} samples, weight={weight:.3f}")
    
    # Build simple CNN (trainable end-to-end)
    print("\nBuilding model (simple CNN)...")
    input_shape = (image_size[0], image_size[1], 1)
    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        model_type="simple",
        dropout=0.4,  # Increased dropout
    )
    
    # Use stronger focal loss
    loss = FocalLoss(alpha=1.0, gamma=3.0)  # gamma=3.0 for stronger focus on hard examples
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    
    # Lower learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    print(f"\nModel summary:")
    model.summary()
    
    # Callbacks
    best_path = out_dir / "cap_classifier_best.keras"
    last_path = out_dir / "cap_classifier.keras"
    class_names_path = out_dir / "class_names.json"
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path.as_posix(),
            monitor="val_acc",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=8,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir.as_posix(),
            histogram_freq=0,
            write_graph=False
        ),
        # Custom callback to print predictions during training
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\nEpoch {epoch+1}: train_acc={logs.get('acc', 0):.4f}, "
                f"val_acc={logs.get('val_acc', 0):.4f}, "
                f"train_loss={logs.get('loss', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}"
            )
        )
    ]
    
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,  # Less verbose output
    )
    
    # Save final model
    model.save(last_path)
    class_names_path.write_text(json.dumps(list(class_names), indent=2), encoding="utf-8")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest model: {best_path}")
    print(f"Last model: {last_path}")
    print(f"Class names: {class_names_path}")
    
    # Quick test evaluation
    print("\n" + "="*80)
    print("QUICK TEST EVALUATION")
    print("="*80 + "\n")
    
    results = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {results[0]:.4f}")
    print(f"Test accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
    
    # Check predictions distribution
    print("\nPrediction distribution on test set:")
    y_pred_all = []
    for x, _ in test_ds:
        p = model.predict(x, verbose=0)
        pred_class = np.argmax(p, axis=1)
        y_pred_all.extend(pred_class.tolist())
    
    y_pred_all = np.array(y_pred_all)
    for i, name in enumerate(class_names):
        count = int((y_pred_all == i).sum())
        pct = 100 * count / len(y_pred_all)
        print(f"  {name:15s}: {count:4d} predictions ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    print("Run full_evaluation.py with v4 model to see detailed metrics")
    print("="*80)


if __name__ == "__main__":
    main()
