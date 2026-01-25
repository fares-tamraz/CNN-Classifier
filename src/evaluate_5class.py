"""Evaluate 5-class model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_datasets
from model import build_model


def load_class_names(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "class_names" in data:
        return list(data["class_names"])
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 5-class model")
    parser.add_argument("--model", "--model_path", dest="model_path", required=True)
    parser.add_argument("--class_names", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--color_mode", type=str, choices=["grayscale", "rgb"], default="grayscale")
    parser.add_argument("--model_type", type=str, choices=["simple", "mobilenetv2"], default="simple")
    args = parser.parse_args()

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    class_names = load_class_names(args.class_names)
    num_classes = len(class_names)
    
    input_channels = 1 if args.color_mode == "grayscale" else 3
    input_shape = (image_size[0], image_size[1], input_channels)
    
    # Build and load model
    model = build_model(
        input_shape=input_shape,
        num_classes=num_classes,
        model_type=args.model_type,
        dropout=0.3
    )
    
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(args.model_path)

    _, _, test_ds, ds_class_names = load_datasets(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode=args.color_mode,
        augment=False,
        seed=42,
        verify_class_order=True,
        verbose=False,
    )

    class_names = list(ds_class_names)

    # Get predictions
    y_true = []
    y_pred = []

    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        pred_class = np.argmax(p, axis=1)
        y_pred.extend(pred_class.tolist())
        
        # Handle both one-hot and sparse labels
        if y.ndim == 2:
            true_class = np.argmax(y.numpy(), axis=1)
        else:
            true_class = y.numpy()
        y_true.extend(true_class.tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # Overall accuracy
    acc = np.mean(y_true == y_pred)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    print("\n=== 5-CLASS TEST RESULTS ===\n")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    
    print("Confusion Matrix:")
    print("(rows=true, cols=predicted)\n")
    header = "".join(f"{c[:8]:>10}" for c in class_names)
    print(f"{'':15}{header}")
    for i, row in enumerate(cm):
        print(f"{class_names[i][:15]:15}" + "".join(f"{v:>10}" for v in row))
    
    print("\n" + "="*80)
    print("\nPer-Class Metrics:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


if __name__ == "__main__":
    main()
