"""Run inference on a single image (or a folder of images) and apply factory-style decisions.

Binary decision logic (recommended for conveyor sorting):
- if confidence >= accept_threshold: trust the prediction
- otherwise: flag for manual review

Example:
  python src/predict.py --model_path models/cap_classifier.keras \
    --class_names models/class_names.json --image path/to/image.jpg

You can also pass --folder to process many images and log results to CSV.
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf


def load_class_names(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["class_names"]


def load_one_image(image_path: Path, image_size, grayscale=True) -> tf.Tensor:
    img = tf.keras.utils.load_img(
        str(image_path),
        target_size=image_size,
        color_mode="grayscale" if grayscale else "rgb",
    )
    arr = tf.keras.utils.img_to_array(img) / 255.0
    arr = tf.expand_dims(arr, 0)
    return arr


def decide_binary(pred_label: str, confidence: float, accept_threshold: float) -> str:
    if confidence >= accept_threshold:
        return "ACCEPT" if pred_label == "good" else "REJECT"
    return "MANUAL_REVIEW"


def write_log_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def predict_path(model, class_names, image_path: Path, image_size, accept_threshold, log_csv):
    t0 = time.perf_counter()
    x = load_one_image(image_path, image_size=image_size, grayscale=True)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = class_names[idx]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    # Decision logic: only meaningful for binary datasets named [faulty, good]
    decision = "N/A"
    if set(class_names) == {"good", "faulty"} and len(class_names) == 2:
        decision = decide_binary(label, conf, accept_threshold)

    print(f"{image_path.name}: {label} (conf={conf:.3f}, {dt_ms:.1f} ms) -> {decision}")

    if log_csv:
        row = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "image": str(image_path),
            "pred_label": label,
            "confidence": f"{conf:.6f}",
            "decision": decision,
            "inference_ms": f"{dt_ms:.3f}",
        }
        write_log_row(log_csv, row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict bottle cap class")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--class_names", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--accept_threshold", type=float, default=0.95)
    parser.add_argument("--log_csv", type=str, default="logs/predictions.csv")

    args = parser.parse_args()

    if (args.image is None) == (args.folder is None):
        raise SystemExit("Provide exactly one of --image or --folder")

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    model = tf.keras.models.load_model(args.model_path)
    class_names = load_class_names(args.class_names)

    log_csv = Path(args.log_csv) if args.log_csv else None

    if args.image:
        predict_path(model, class_names, Path(args.image), image_size, args.accept_threshold, log_csv)
        return

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(folder)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in exts:
            predict_path(model, class_names, p, image_size, args.accept_threshold, log_csv)


if __name__ == "__main__":
    main()
