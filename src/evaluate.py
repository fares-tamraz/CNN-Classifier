"""Evaluate a trained model on the test set.

For the binary model in this repo (sigmoid output), we report:
- Accuracy
- Confusion matrix
- Precision/Recall/F1 for BOTH classes

Usage:
  python src/evaluate.py --model models/cap_classifier_best.keras \
    --class_names models/class_names.json --data_dir data/processed/cls_binary_fullframe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

# When running as a script from repo root, `src` is a package.


def load_class_names(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "class_names" in data:
        return list(data["class_names"])
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a bottle cap classifier")
    parser.add_argument("--model", "--model_path", dest="model_path", required=True)
    parser.add_argument("--class_names", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--color_mode", type=str, choices=["grayscale", "rgb"], default="grayscale")
    parser.add_argument("--threshold", type=float, default=0.5, help="p_good threshold for label")
    args = parser.parse_args()

    # Local import to avoid path issues
    from src.data_loader import load_datasets

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    model = tf.keras.models.load_model(args.model_path)
    class_names = load_class_names(args.class_names)

    _, _, test_ds, ds_class_names = load_datasets(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode=args.color_mode,
        augment=False,
        seed=42,
        verify_classes=True,
        verbose=False,
    )

    if list(ds_class_names) != list(class_names):
        print("Warning: class_names.json differs from dataset class order.")
        print("  class_names.json:", class_names)
        print("  dataset order:   ", ds_class_names)
        class_names = list(ds_class_names)

    # Predict
    y_true = []
    y_prob = []

    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        p = np.array(p)
        if p.ndim == 2 and p.shape[1] == 1:
            y_prob.extend(p[:, 0].tolist())
        elif p.ndim == 2 and p.shape[1] == 2:
            # softmax
            good_idx = class_names.index("good")
            y_prob.extend(p[:, good_idx].tolist())
        else:
            raise ValueError(f"Unexpected model output shape: {p.shape}")

        y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    # Determine which index corresponds to good
    if len(class_names) != 2 or set(class_names) != {"faulty", "good"}:
        raise SystemExit(f"Expected binary classes {{faulty,good}}. Got: {class_names}")

    good_idx = class_names.index("good")
    # For binary labels produced by image_dataset_from_directory, good is usually 1
    # but we handle either ordering.
    if good_idx == 1:
        y_true_good = y_true
    else:
        y_true_good = 1 - y_true

    y_pred_good = (y_prob >= args.threshold).astype(int)

    # Confusion matrix w.r.t good as positive
    tp = int(np.sum((y_true_good == 1) & (y_pred_good == 1)))
    fp = int(np.sum((y_true_good == 0) & (y_pred_good == 1)))
    fn = int(np.sum((y_true_good == 1) & (y_pred_good == 0)))
    tn = int(np.sum((y_true_good == 0) & (y_pred_good == 0)))

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    def prf(tp_, fp_, fn_):
        prec = tp_ / max(1, tp_ + fp_)
        rec = tp_ / max(1, tp_ + fn_)
        f1 = (2 * prec * rec) / max(1e-12, prec + rec)
        return prec, rec, f1

    good_prec, good_rec, good_f1 = prf(tp, fp, fn)
    # Faulty is the negative class here
    faulty_tp = tn
    faulty_fp = fn
    faulty_fn = fp
    faulty_prec, faulty_rec, faulty_f1 = prf(faulty_tp, faulty_fp, faulty_fn)

    print("\n=== Test results ===")
    print(f"threshold(p_good) = {args.threshold:.3f}")
    print(f"accuracy          = {acc:.4f}")
    print("\nConfusion matrix (good=positive):")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")

    print("\nPer-class metrics:")
    print(f"  GOOD   prec={good_prec:.3f} rec={good_rec:.3f} f1={good_f1:.3f}")
    print(f"  FAULTY prec={faulty_prec:.3f} rec={faulty_rec:.3f} f1={faulty_f1:.3f}")


if __name__ == "__main__":
    main()
