"""Evaluate a trained model on the test set.

Outputs:
- accuracy
- confusion matrix
- precision/recall/F1 (classification report)

Example:
  python src/evaluate.py --data_dir data/processed/cls_binary_crops \
    --model_path models/cap_classifier.keras --class_names models/class_names.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from .data_loader import load_datasets


def load_class_names(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["class_names"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate classifier")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--class_names", type=str, required=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    _, _, test_ds, _ = load_datasets(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode="grayscale",
        augment=False,
        seed=42,
    )

    class_names = load_class_names(args.class_names)

    model = tf.keras.models.load_model(args.model_path)

    y_true = []
    y_pred = []

    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred)

    print(f"Test accuracy: {acc:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


if __name__ == "__main__":
    main()
