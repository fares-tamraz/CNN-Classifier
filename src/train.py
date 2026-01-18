"""Train a classifier on the prepared dataset.

Recommended workflow:
1) Place Roboflow YOLO dataset in data/raw/<...>
2) Run tools/prepare_dataset.py to generate a processed dataset
3) Train with this script

Example:
  python src/train.py --data_dir data/processed/cls_binary_crops --epochs 20 --augment
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from data_loader import load_datasets
from model import build_model


def compute_class_weights(train_ds: tf.data.Dataset, num_classes: int) -> Dict[int, float]:
    """Compute inverse-frequency class weights for imbalanced datasets."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_ds.unbatch():
        counts[int(y.numpy())] += 1

    total = counts.sum()
    weights = {}
    for i in range(num_classes):
        if counts[i] == 0:
            weights[i] = 1.0
        else:
            # classic balanced weight
            weights[i] = float(total / (num_classes * counts[i]))
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN classifier")
    parser.add_argument("--data_dir", type=str, required=True, help="Processed dataset folder (has train/val/test)")
    parser.add_argument("--model_out", type=str, default="models/cap_classifier.keras")
    parser.add_argument("--class_names_out", type=str, default="models/class_names.json")

    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model_type", choices=["simple", "mobilenetv2"], default="simple")
    parser.add_argument("--augment", action="store_true", help="Enable mild data augmentation")
    parser.add_argument("--use_class_weights", action="store_true", help="Compute class weights from training set")

    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--checkpoint", type=str, default="models/best.keras")

    args = parser.parse_args()

    # Reproducibility
    tf.keras.utils.set_random_seed(args.seed)

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    train_ds, val_ds, _, class_names = load_datasets(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode="grayscale",
        augment=args.augment,
        seed=args.seed,
    )

    num_classes = len(class_names)
    input_shape = (image_size[0], image_size[1], 1)

    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        model_type=args.model_type,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stop_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=args.checkpoint,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(train_ds, num_classes=num_classes)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save final model + class names
    model.save(args.model_out)

    Path(args.class_names_out).write_text(
        json.dumps({"class_names": class_names}, indent=2),
        encoding="utf-8",
    )

    # Save training history (optional, but useful)
    hist_out = Path(args.model_out).with_suffix(".history.json")
    hist_out.write_text(json.dumps(history.history, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Saved model: {Path(args.model_out).resolve()}")
    print(f"Saved class names: {Path(args.class_names_out).resolve()}")


if __name__ == "__main__":
    main()
