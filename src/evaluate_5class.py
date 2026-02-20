"""Evaluate multi-class (5-class or 3-class) model.

Usage:
  python src/evaluate_5class.py \
    --model models/v4/cap_classifier_best.keras \
    --class_names models/v4/class_names.json \
    --data_dir data/processed/cls_5class_crops_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from data_loader import load_datasets


def load_class_names(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "class_names" in data:
        return list(data["class_names"])
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-class model")
    parser.add_argument("--model", "--model_path", dest="model_path", required=True)
    parser.add_argument("--class_names", type=str, default=None, help="Optional: path to class_names.json (unused, class order inferred from dataset)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--color_mode", type=str, choices=["grayscale", "rgb"], default="grayscale")
    args = parser.parse_args()

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    # Load full saved model — handles custom layers (Lambda, FocalLoss) gracefully
    try:
        model = tf.keras.models.load_model(args.model_path, compile=False)
    except (NotImplementedError, ValueError) as e:
        print(f"Note: Standard load failed ({type(e).__name__}), attempting workaround...")
        model = tf.keras.models.load_model(args.model_path, safe_mode=False, compile=False)

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
    num_classes = len(class_names)

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
