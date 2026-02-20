#!/usr/bin/env python3
"""
Train v6 model on 3-class dataset - FIXED version.

Fixes from v5:
1. Augmentation in data pipeline, NOT in model (fixes validation instability)
2. Lower learning rate (5e-4 → 1e-4)
3. Lower patience (12 → 6)
4. Removed cosine annealing (simpler, more stable)
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_loader import load_datasets


# ============================================================================
# FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================================
class FocalLoss(tf.keras.losses.Loss):
    """Focal Loss with optional label smoothing."""
    
    def __init__(self, alpha=1.0, gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), num_classes)
        y_true_onehot = tf.cast(y_true_onehot, y_pred.dtype)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            y_true_onehot = y_true_onehot * (1 - self.label_smoothing) + \
                           self.label_smoothing / tf.cast(num_classes, y_pred.dtype)
        
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce = -y_true_onehot * tf.math.log(y_pred)
        ce = tf.reduce_sum(ce, axis=-1)
        
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        return tf.reduce_mean(self.alpha * focal_weight * ce)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "label_smoothing": self.label_smoothing
        })
        return config


# ============================================================================
# MODEL ARCHITECTURE (NO AUGMENTATION - moved to data pipeline)
# ============================================================================
def build_simple_cnn(input_shape, num_classes, dropout=0.4):
    """Build simple CNN - augmentation handled separately in data pipeline."""
    
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
    
    model = tf.keras.Model(inputs, outputs)
    return model


# ============================================================================
# CLASS WEIGHTS
# ============================================================================
def compute_class_weights(y_train: np.ndarray, boost_minority: float = 1.5) -> dict:
    """Compute balanced class weights with optional minority boosting."""
    classes, counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(classes)
    
    weights = {}
    for cls, count in zip(classes, counts):
        weight = n_samples / (n_classes * count)
        weights[int(cls)] = weight
    
    # Boost minority class weights
    max_weight = max(weights.values())
    for cls in weights:
        if weights[cls] > 1.0:
            weights[cls] = min(weights[cls] * boost_minority, max_weight * 2)
    
    return weights


# ============================================================================
# MAIN TRAINING
# ============================================================================
def main():
    print("="*70)
    print("V6 3-CLASS MODEL TRAINING (FIXED)")
    print("="*70)
    
    # Configuration
    data_dir = "data/processed/cls_3class_crops"
    out_dir = Path("models/v6")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    image_size = (224, 224)
    batch_size = 32
    epochs = 50
    initial_lr = 1e-4  # Lower LR
    dropout = 0.4
    label_smoothing = 0.1
    focal_gamma = 2.5
    patience = 6  # Lower patience
    
    print(f"\nConfiguration:")
    print(f"  Data: {data_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Image size: {image_size}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Dropout: {dropout}")
    print(f"  Label smoothing: {label_smoothing}")
    print(f"  Focal gamma: {focal_gamma}")
    print(f"  Patience: {patience}")
    
    # Load datasets - AUGMENTATION IN DATA PIPELINE
    print("\nLoading datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        augment=True,  # Augmentation in data pipeline, not model
    )
    
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    
    # Save class names
    class_names_path = out_dir / "class_names.json"
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    
    # Extract labels for class weights
    y_train = []
    for _, labels in train_ds:
        y_train.extend(labels.numpy())
    y_train = np.array(y_train)
    
    class_weights = compute_class_weights(y_train, boost_minority=1.5)
    print(f"\nClass weights:")
    for idx, name in enumerate(class_names):
        print(f"  {name}: {class_weights[idx]:.3f}")
    
    # Build model
    print("\nBuilding model...")
    input_shape = (*image_size, 1)  # Grayscale
    model = build_simple_cnn(input_shape, num_classes, dropout=dropout)
    
    # Compile
    loss = FocalLoss(alpha=1.0, gamma=focal_gamma, label_smoothing=label_smoothing)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=initial_lr, weight_decay=1e-5)
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )
    
    model.summary()
    
    # Callbacks
    best_path = out_dir / "cap_classifier_best.keras"
    log_dir = out_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path.as_posix(),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
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
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"\n>>> Epoch {epoch+1}: "
                f"train_acc={logs.get('accuracy', 0):.4f}, "
                f"val_acc={logs.get('val_accuracy', 0):.4f}, "
                f"train_loss={logs.get('loss', 0):.4f}, "
                f"val_loss={logs.get('val_loss', 0):.4f}"
            )
        )
    ]
    
    # Train
    print("\n" + "="*70)
    print("TRAINING STARTED")
    print("="*70)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    final_path = out_dir / "cap_classifier_final.keras"
    model.save(final_path.as_posix())
    
    # Print final results
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    best_epoch = history.history["val_accuracy"].index(best_val_acc) + 1
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"\nModels saved to:")
    print(f"  Best: {best_path}")
    print(f"  Final: {final_path}")
    print(f"  Class names: {class_names_path}")
    print(f"  TensorBoard logs: {log_dir}")
    
    # Quick test evaluation
    print("\n" + "-"*70)
    print("TEST SET EVALUATION")
    print("-"*70)
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")


if __name__ == "__main__":
    main()
