"""Quick sanity-check for dataset + model:

- loads a split (val by default)
- runs predictions
- prints confusion matrix + a few worst mistakes
- optionally writes misclassified images into tools/_debug/misclassified

Usage:
  python tools/debug_val_preds.py --model models/cap_classifier_best.keras \
    --data_dir data/processed/cls_binary_fullframe --split val --save_images
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_datasets


def sigmoid_to_p_good(y_pred: np.ndarray) -> np.ndarray:
    y_pred = np.asarray(y_pred)
    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        return y_pred[:, 0]
    if y_pred.ndim == 2 and y_pred.shape[1] == 2:
        # softmax
        return y_pred[:, 1]
    raise ValueError(f"Unexpected prediction shape: {y_pred.shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/cap_classifier_best.keras")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--save_images", action="store_true")
    ap.add_argument("--max_images", type=int, default=25)
    ap.add_argument("--th", type=float, default=0.5)
    args = ap.parse_args()

    model = tf.keras.models.load_model(args.model)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data_dir,
        batch_size=args.batch_size,
        color_mode="grayscale",
        augment=False,
        verify_class_order=True,
        verbose=False,
    )

    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split]

    y_true = []
    y_pred = []
    x_batches = []
    y_batches = []

    for xb, yb in ds:
        yp = model.predict(xb, verbose=0)
        p_good = sigmoid_to_p_good(yp)
        yhat = (p_good >= args.th).astype(np.int32)
        y_true.append(yb.numpy().astype(np.int32))
        y_pred.append(yhat)
        if args.save_images and len(x_batches) < args.max_images:
            x_batches.append(xb.numpy())
            y_batches.append(yb.numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Confusion matrix (rows=true, cols=pred)
    # class 0=faulty, 1=good (expected)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    acc = (y_true == y_pred).mean()
    print(f"\nSplit={args.split} | th={args.th:.2f} | acc={acc:.3f}")
    print("Confusion matrix (rows=true [faulty, good], cols=pred [faulty, good])")
    print(cm)

    if not args.save_images:
        return

    out_dir = ROOT / "tools" / "_debug" / "misclassified"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for xb, yb in ds.take(9999):
        yp = model.predict(xb, verbose=0)
        p_good = sigmoid_to_p_good(yp)
        yhat = (p_good >= args.th).astype(np.int32)
        for i in range(xb.shape[0]):
            if int(yb[i]) == int(yhat[i]):
                continue
            img = xb[i].numpy()
            # img is (H,W,1) float [0,1]
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
            pil = Image.fromarray(img_u8[:, :, 0], mode="L")
            true_name = class_names[int(yb[i])]
            pred_name = class_names[int(yhat[i])]
            fname = f"true_{true_name}__pred_{pred_name}__p_good_{p_good[i]:.3f}__{saved:03d}.png"
            pil.save(out_dir / fname)
            saved += 1
            if saved >= args.max_images:
                print(f"Saved {saved} misclassified images to: {out_dir}")
                return

    print(f"Saved {saved} misclassified images to: {out_dir}")


if __name__ == "__main__":
    main()
