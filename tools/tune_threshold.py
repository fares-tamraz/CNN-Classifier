"""Tune a probability threshold for the cap classifier.

Sweeps thresholds on p(good) and prints accuracy / precision / recall / F1
for both the 'good' and 'faulty' classes.  Works with 2-class and 3-class
(faulty / good / no_cap) models — for the 3-class model, threshold is applied
to the 'good' softmax probability; anything below the threshold is treated as
a reject (faulty or no_cap).

Usage:
  python tools/tune_threshold.py \
    --model models/v7/cap_classifier_best.keras \
    --class_names models/v7/class_names.json \
    --data_dir data/processed/cls_3class_crops --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Allow `python tools/tune_threshold.py` from anywhere
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_datasets
from src.predict import load_class_names


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return confusion + precision/recall/f1 for good and faulty."""
    # y: 0=faulty,1=good
    tp_good = int(((y_true == 1) & (y_pred == 1)).sum())
    fp_good = int(((y_true == 0) & (y_pred == 1)).sum())
    fn_good = int(((y_true == 1) & (y_pred == 0)).sum())
    tn_good = int(((y_true == 0) & (y_pred == 0)).sum())

    # For faulty-as-positive, swap roles
    tp_faulty = tn_good
    fp_faulty = fn_good
    fn_faulty = fp_good
    tn_faulty = tp_good

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    good_p, good_r, good_f1 = prf(tp_good, fp_good, fn_good)
    faulty_p, faulty_r, faulty_f1 = prf(tp_faulty, fp_faulty, fn_faulty)
    acc = (tp_good + tn_good) / len(y_true) if len(y_true) else 0.0

    return {
        "acc": acc,
        "good_prec": good_p,
        "good_rec": good_r,
        "good_f1": good_f1,
        "faulty_prec": faulty_p,
        "faulty_rec": faulty_r,
        "faulty_f1": faulty_f1,
        "tp_good": tp_good,
        "fp_good": fp_good,
        "fn_good": fn_good,
        "tn_good": tn_good,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .keras model")
    ap.add_argument("--class_names", default="models/v7/class_names.json")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--steps", type=int, default=19, help="Number of thresholds between 0.05 and 0.95")
    args = ap.parse_args()

    try:
        model = tf.keras.models.load_model(args.model, compile=False)
    except (NotImplementedError, ValueError):
        model = tf.keras.models.load_model(args.model, compile=False, safe_mode=False)
    class_names = load_class_names(args.class_names)

    train_ds, val_ds, test_ds, ds_class_names = load_datasets(
        args.data_dir,
        color_mode="grayscale",
        augment=False,
        shuffle_val=False,
        verbose=True,
    )

    if list(ds_class_names) != list(class_names):
        print(f"WARNING: dataset class order {ds_class_names} != class_names.json {class_names}")

    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    y_true = []
    p_good = []

    for x, y in ds:
        y = y.numpy().astype(int)
        y_true.append(y)

        # batch inference
        yhat = model.predict(x, verbose=0)
        yhat = np.array(yhat)

        if yhat.ndim == 2 and yhat.shape[1] == 1:
            # sigmoid — map to p_good
            p = yhat[:, 0]
            if len(class_names) == 2 and class_names[1] != "good":
                p = 1.0 - p
            p_good.append(p)
        elif yhat.ndim == 2 and yhat.shape[1] >= 2:
            # softmax — binary or N-class: extract the 'good' column
            good_idx = class_names.index("good") if "good" in class_names else yhat.shape[1] - 1
            p_good.append(yhat[:, good_idx])
        else:
            raise ValueError(f"Unexpected output shape: {yhat.shape}")

    y_true_raw = np.concatenate(y_true)
    p_good = np.concatenate(p_good)

    # Binarize: 1=good, 0=everything-else (faulty / no_cap)
    good_idx = class_names.index("good") if "good" in class_names else 1
    y_true = (y_true_raw == good_idx).astype(int)

    print(f"\nSplit: {args.split} | N={len(y_true)}")
    print(f"Classes (from file): {class_names}")
    print(f"Good index: {good_idx}  |  Good count: {y_true.sum()}  |  Reject count: {(y_true==0).sum()}")

    thresholds = np.linspace(0.05, 0.95, args.steps)

    print("\nThresh   Acc     GoodP   GoodR   GoodF1  FaultyP FaultyR FaultyF1")
    best = (None, -1.0, None)  # (t, score, metrics)
    for t in thresholds:
        y_pred = (p_good >= t).astype(int)
        m = compute_metrics(y_true, y_pred)
        print(
            f"{t:0.2f}    {m['acc']:.3f}   {m['good_prec']:.3f}  {m['good_rec']:.3f}  {m['good_f1']:.3f}   "
            f"{m['faulty_prec']:.3f}  {m['faulty_rec']:.3f}  {m['faulty_f1']:.3f}"
        )
        # Optimize for faulty F1 (common for safety) but you can change this
        if m["faulty_f1"] > best[1]:
            best = (t, m["faulty_f1"], m)

    t_best, score, m_best = best
    print(
        f"\nBest by Faulty-F1: th={t_best:0.2f} | acc={m_best['acc']:.3f} "
        f"faulty_prec={m_best['faulty_prec']:.3f} faulty_rec={m_best['faulty_rec']:.3f}"
    )


if __name__ == "__main__":
    main()
