"""Two-stage inference.

Stage A (binary): good vs faulty
Stage B (fault-type): broken_cap / broken_ring / loose_cap / no_cap

This matches a realistic factory flow:
- first decide PASS/FAIL
- if FAIL, then classify the failure reason (for statistics + maintenance)

Example:
  python src/predict_two_stage.py \
    --stageA_model models/stageA_binary.keras \
    --stageA_classes models/stageA_classes.json \
    --stageB_model models/stageB_faulttype.keras \
    --stageB_classes models/stageB_classes.json \
    --image path/to/image.jpg
"""

import argparse
import json
import time
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


def predict(model, x):
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return idx, conf


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage bottle cap inspection")
    parser.add_argument("--stageA_model", type=str, required=True)
    parser.add_argument("--stageA_classes", type=str, required=True)
    parser.add_argument("--stageB_model", type=str, required=True)
    parser.add_argument("--stageB_classes", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--accept_threshold", type=float, default=0.95)

    args = parser.parse_args()

    image_size = (int(args.image_size[0]), int(args.image_size[1]))
    image_path = Path(args.image)

    stageA = tf.keras.models.load_model(args.stageA_model)
    stageB = tf.keras.models.load_model(args.stageB_model)

    classesA = load_class_names(args.stageA_classes)
    classesB = load_class_names(args.stageB_classes)

    if set(classesA) != {"good", "faulty"}:
        raise ValueError("Stage A must be trained on binary classes: good/faulty")

    x = load_one_image(image_path, image_size=image_size, grayscale=True)

    t0 = time.perf_counter()
    idxA, confA = predict(stageA, x)
    labelA = classesA[idxA]

    if confA < args.accept_threshold:
        print(f"StageA: {labelA} (conf={confA:.3f}) -> MANUAL_REVIEW")
        return

    if labelA == "good":
        print(f"StageA: good (conf={confA:.3f}) -> ACCEPT")
        return

    # Stage B only runs for confident FAIL
    idxB, confB = predict(stageB, x)
    labelB = classesB[idxB]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    print(f"StageA: faulty (conf={confA:.3f}) -> REJECT")
    print(f"StageB: {labelB} (conf={confB:.3f})")
    print(f"Total inference: {dt_ms:.1f} ms")


if __name__ == "__main__":
    main()
