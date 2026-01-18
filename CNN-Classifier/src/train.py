"""Train a classifier for bottle-cap inspection.

Typical workflow:
1) Convert YOLO detection dataset -> classification dataset
   python tools/prepare_dataset.py --yolo_root data/raw/bottle-cap-yolo \
     --mode binary --style fullframe --out_dir data/processed --overwrite

2) Train (MobileNetV2 recommended)
   python src/train.py --data_dir data/processed/cls_binary_fullframe \
     --model_type mobilenetv2 --epochs 20 --augment

Models are saved into --out_dir (default: models).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from data_loader import load_datasets
from model import build_model


def compute_class_weights(y: np.ndarray) -> dict[int, float]:
    """Balanced weights: total/(num_classes*count_c)."""
    classes, counts = np.unique(y, return_counts=True)
    n = len(y)
    k = len(classes)
    weights = {int(c): float(n / (k * cnt)) for c, cnt in zip(classes, counts)}
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bottle cap classifier")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="models")

    parser.add_argument("--model_type", type=str, default="mobilenetv2", choices=["simple_cnn", "mobilenetv2"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--color_mode", type=str, default="grayscale", choices=["grayscale", "rgb"])
    parser.add_argument("--augment", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--fine_tune", action="store_true", help="Unfreeze backbone for a short fine-tune stage")
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)

    args = parser.parse_args()

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode=args.color_mode,
        augment=args.augment,
        seed=args.seed,
        verbose=True,
        verify_class_order=True,
    )

    num_classes = len(class_names)
    if num_classes < 2:
        raise SystemExit(f"Need at least 2 classes. Found: {class_names}")

    input_channels = 1 if args.color_mode == "grayscale" else 3
    input_shape = (image_size[0], image_size[1], input_channels)

    model = build_model(model_type=args.model_type, num_classes=num_classes, input_shape=input_shape)

    # --- Loss/metrics match the model output ---
    is_binary_sigmoid = (num_classes == 2 and model.output_shape[-1] == 1)

    if is_binary_sigmoid:
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss=loss, metrics=metrics)

    # --- Class weights (helps imbalance) ---
    y_train = np.concatenate([y.numpy() for _, y in train_ds.unbatch().batch(4096)])
    class_weights = compute_class_weights(y_train)
    print(f"Train counts: { {class_names[int(c)]: int((y_train==c).sum()) for c in sorted(class_weights)} }")
    print(f"Class weights: {class_weights}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save BOTH "best" and "last" for convenience
    best_path = out_dir / "cap_classifier_best.keras"
    last_path = out_dir / "cap_classifier.keras"
    class_names_path = out_dir / "class_names.json"

    monitor = "val_auc" if any(m.name == "auc" for m in metrics) else "val_acc"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(best_path.as_posix(), monitor=monitor, save_best_only=True, mode="max"),
        tf.keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, mode="max", factor=0.5, patience=2, min_lr=1e-6),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Optional fine-tuning (MobileNetV2 only)
    if args.fine_tune and args.model_type == "mobilenetv2":
        # Unfreeze backbone (the first large layer is usually the base model)
        for layer in model.layers:
            layer.trainable = True
        model.compile(optimizer=tf.keras.optimizers.Adam(args.fine_tune_lr), loss=loss, metrics=metrics)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1,
        )

    # Save last model
    model.save(last_path)

    # Standardize to a simple JSON list for zero-confusion
    class_names_path.write_text(json.dumps(list(class_names), indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Saved last model: {last_path}")
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved class names: {class_names_path}")

    # Quick test-set evaluation (sanity)
    try:
        results = model.evaluate(test_ds, verbose=0)
        names = model.metrics_names
        print("Test metrics:")
        for n, v in zip(names, results):
            print(f"  {n}: {v:.4f}")
    except Exception as e:
        print(f"(Test evaluation skipped: {e})")


if __name__ == "__main__":
    main()
