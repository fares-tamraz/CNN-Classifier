"""Find optimal threshold to minimize false positives or maximize F1 score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

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
    parser = argparse.ArgumentParser(description="Find optimal threshold")
    parser.add_argument("--model", "--model_path", dest="model_path", required=True)
    parser.add_argument("--class_names", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--color_mode", type=str, choices=["grayscale", "rgb"], default="grayscale")
    parser.add_argument("--model_type", type=str, choices=["simple", "mobilenetv2"], default="mobilenetv2")
    args = parser.parse_args()

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    class_names = load_class_names(args.class_names)
    num_classes = len(class_names)
    
    input_channels = 1 if args.color_mode == "grayscale" else 3
    input_shape = (image_size[0], image_size[1], input_channels)
    
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
    y_prob = []

    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        p = np.array(p)
        good_idx = class_names.index("good")
        if p.ndim == 2 and p.shape[1] == 2:
            y_prob.extend(p[:, good_idx].tolist())
        else:
            raise ValueError(f"Unexpected shape: {p.shape}")
        y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    good_idx = class_names.index("good")
    if good_idx == 1:
        y_true_good = y_true
    else:
        y_true_good = 1 - y_true

    # Sweep thresholds
    print("\n=== Threshold Analysis ===\n")
    print(f"{'Threshold':<12} {'Acc':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'FP_Rate':<10} {'F1_good':<10}")
    print("-" * 80)

    best_threshold = 0.5
    best_f1 = 0
    best_acc = 0
    min_fp = float('inf')

    for threshold in np.linspace(0.1, 0.9, 17):
        y_pred_good = (y_prob >= threshold).astype(int)

        tp = int(np.sum((y_true_good == 1) & (y_pred_good == 1)))
        fp = int(np.sum((y_true_good == 0) & (y_pred_good == 1)))
        fn = int(np.sum((y_true_good == 1) & (y_pred_good == 0)))
        tn = int(np.sum((y_true_good == 0) & (y_pred_good == 0)))

        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-12, prec + rec)
        fp_rate = fp / max(1, fp + tn)

        print(f"{threshold:<12.2f} {acc:<8.4f} {tp:<6} {fp:<6} {fn:<6} {tn:<6} {fp_rate:<10.4f} {f1:<10.4f}")

        if fp < min_fp:
            min_fp = fp
            best_threshold = threshold
            best_f1 = f1
            best_acc = acc

    print("\n" + "=" * 80)
    print(f"\nBest threshold for minimizing false positives: {best_threshold:.2f}")
    print(f"  Accuracy: {best_acc:.4f}, F1 (good): {best_f1:.4f}, FP count: {int(min_fp)}")


if __name__ == "__main__":
    main()
