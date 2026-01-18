"""Predict a bottle cap class for a single image or a folder.

This project is designed for "factory-style" sorting, so we avoid a single hard
threshold. Instead we use an *accept / reject / manual_review* band:

- ACCEPT if p_good >= th_accept
- REJECT if p_good <= th_reject
- MANUAL_REVIEW otherwise

Why? Because in real inspection you usually want:
- very low false accepts of bad parts (high faulty recall), and
- a controlled "unsure" stream for re-check / manual inspection.

Examples:
  # single image
  python src/predict.py --model models/cap_classifier_best.keras \
    --class_names models/class_names.json \
    --image "data/processed/cls_binary_fullframe/test/good/some.png" \
    --th_accept 0.60 --th_reject 0.40

  # folder + CSV log
  python src/predict.py --model models/cap_classifier_best.keras \
    --class_names models/class_names.json \
    --folder data/processed/cls_binary_fullframe/test \
    --th_accept 0.60 --th_reject 0.40 --log_csv logs/preds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_class_names(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # supports either ["faulty","good"] OR {"class_names":[...]}
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "class_names" in data:
        return [str(x) for x in data["class_names"]]
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


def load_one_image(image_path: Path, image_size: tuple[int, int], *, color_mode: str) -> tf.Tensor:
    """Loads and returns shape (1,H,W,C) float32 in [0,1]."""
    img = tf.keras.utils.load_img(
        str(image_path),
        target_size=image_size,
        color_mode=color_mode,
    )
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = tf.expand_dims(arr, 0)
    return arr


def _as_numpy(y) -> np.ndarray:
    y = np.array(y)
    # Some keras versions return a list
    if isinstance(y, list):
        y = np.array(y)
    return y


def infer_p_good(model: tf.keras.Model, class_names: list[str], x: tf.Tensor) -> float:
    """Returns p(good) for either sigmoid (1 output) or softmax (2 outputs)."""
    y = _as_numpy(model.predict(x, verbose=0))

    # sigmoid binary: (1,1) => probability of the *positive* class (index 1)
    if y.ndim == 2 and y.shape[1] == 1:
        p_pos = float(y[0, 0])
        # class order matters: Keras binary uses label 1 as "positive".
        # We want p(good). If class_names[1] is good => p_good = p_pos.
        if len(class_names) == 2 and class_names[1] == "good":
            return p_pos
        if len(class_names) == 2 and class_names[1] == "faulty":
            return 1.0 - p_pos
        # fallback: assume ['faulty','good'] (most common)
        return p_pos

    # softmax binary: (1,2)
    if y.ndim == 2 and y.shape[1] == 2:
        probs = y[0]
        if "good" in class_names:
            return float(probs[class_names.index("good")])
        return float(probs[1])  # common default

    raise ValueError(f"Unexpected model output shape: {y.shape}")


def decide_from_band(p_good: float, *, th_accept: float, th_reject: float) -> str:
    if p_good >= th_accept:
        return "ACCEPT"
    if p_good <= th_reject:
        return "REJECT"
    return "MANUAL_REVIEW"


@dataclass
class PredRow:
    image: str
    p_good: float
    decision: str
    inference_ms: float


def write_log_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def iter_images(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS:
            yield p


def predict_one(
    model: tf.keras.Model,
    class_names: list[str],
    image_path: Path,
    *,
    image_size: tuple[int, int],
    th_accept: float,
    th_reject: float,
    color_mode: str,
) -> PredRow:
    t0 = time.perf_counter()
    x = load_one_image(image_path, image_size=image_size, color_mode=color_mode)
    p_good = infer_p_good(model, class_names, x)
    decision = decide_from_band(p_good, th_accept=th_accept, th_reject=th_reject)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return PredRow(image=str(image_path), p_good=p_good, decision=decision, inference_ms=dt_ms)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict bottle cap class for an image or folder")

    # Backwards compatible flags (you used --model in PowerShell)
    parser.add_argument("--model", type=str, default=None, help="Path to .keras model")
    parser.add_argument("--model_path", type=str, default=None, help="(alias of --model)")

    parser.add_argument("--class_names", type=str, default="models/class_names.json")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)

    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])

    # Threshold band
    parser.add_argument("--th_accept", type=float, default=0.60, help="ACCEPT if p_good >= this")
    parser.add_argument("--th_reject", type=float, default=0.40, help="REJECT if p_good <= this")

    # Old name still supported
    parser.add_argument("--accept_threshold", type=float, default=None, help="(deprecated) sets th_accept=th_reject")

    parser.add_argument("--color_mode", choices=["grayscale", "rgb"], default="grayscale")
    parser.add_argument("--log_csv", type=str, default="logs/predictions.csv")

    args = parser.parse_args()

    model_path = args.model or args.model_path
    if not model_path:
        raise SystemExit("Provide --model (or --model_path)")

    if (args.image is None) == (args.folder is None):
        raise SystemExit("Provide exactly one of --image or --folder")

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    if args.accept_threshold is not None:
        args.th_accept = float(args.accept_threshold)
        args.th_reject = float(args.accept_threshold)

    if args.th_reject > args.th_accept:
        raise SystemExit("Invalid thresholds: th_reject must be <= th_accept")

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(args.class_names)

    # guard for binary projects
    if set(class_names) >= {"faulty", "good"} and len(class_names) == 2:
        pass

    log_csv = Path(args.log_csv) if args.log_csv else None

    target = Path(args.image) if args.image else Path(args.folder)
    for img_path in iter_images(target):
        row = predict_one(
            model,
            class_names,
            img_path,
            image_size=image_size,
            th_accept=args.th_accept,
            th_reject=args.th_reject,
            color_mode=args.color_mode,
        )

        name = Path(row.image).name
        print(f"{name}: p_good={row.p_good:.3f} -> {row.decision} ({row.inference_ms:.1f} ms)")

        if log_csv:
            log = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "image": row.image,
                "p_good": f"{row.p_good:.6f}",
                "decision": row.decision,
                "inference_ms": f"{row.inference_ms:.3f}",
            }
            write_log_row(log_csv, log)


if __name__ == "__main__":
    main()
