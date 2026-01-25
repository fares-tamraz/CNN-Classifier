"""Real-time webcam sorter demo.

Controls:
  SPACE = classify current frame
  Q     = quit

Decision policy (factory-style):
  - ACCEPT if p_good >= th_accept
  - REJECT if p_good <= th_reject
  - otherwise -> UNSURE

If UNSURE, we (optionally) "recheck" by reversing the belt briefly (simulated
here) and taking a second sample. Final decision is based on the average.

**Dry run by default:** prints [BELT] / [CHUTE] only; no hardware. Use
--no_dry_run when connected to real servos. Use --show_roi to see the crop.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf


def load_class_names(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    # supports either ["faulty","good"] OR {"class_names":[...]}
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "class_names" in data:
        return [str(x) for x in data["class_names"]]
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


def _infer_p_good_from_output(y: np.ndarray, class_names: list[str]) -> float:
    """Supports sigmoid (shape (1,1)) and softmax (shape (1,2))."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] == 1:
        p_pos = float(y[0, 0])
        # In binary training with class_names ['faulty','good'], index 1 is 'good'
        if len(class_names) == 2 and class_names[1] == "good":
            return p_pos
        if len(class_names) == 2 and class_names[1] == "faulty":
            return 1.0 - p_pos
        # fallback: assume p_pos is p_good
        return p_pos

    if y.ndim == 2 and y.shape[1] == 2:
        probs = y[0]
        if "good" in class_names:
            return float(probs[class_names.index("good")])
        # fallback: assume index 1 is good
        return float(probs[1])

    raise ValueError(f"Unexpected model output shape: {y.shape}")


def preprocess_frame(frame_bgr: np.ndarray, image_size: Tuple[int, int], roi: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop ROI, convert to grayscale, resize, normalize.

    Returns float32 array shape (1,H,W,C) where C matches model input channels.
    """
    h, w = frame_bgr.shape[:2]
    rx1, rx2, ry1, ry2 = roi
    x1 = int(w * rx1)
    x2 = int(w * rx2)
    y1 = int(h * ry1)
    y2 = int(h * ry2)

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    crop = frame_bgr[y1:y2, x1:x2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)

    x = gray.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=(0, -1))  # (1,H,W,1)
    return x


def match_model_channels(x: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """Ensure x has the right channel count for the loaded model."""
    in_shape = model.input_shape
    # Could be list for multi-input models; we only support single input here.
    if isinstance(in_shape, list):
        in_shape = in_shape[0]

    ch = in_shape[-1]
    if ch == 1:
        return x
    if ch == 3:
        return np.repeat(x, repeats=3, axis=-1)
    raise ValueError(f"Unsupported model input channels: {ch}")


def decide_with_recheck(
    p_good_1: float,
    p_good_2: Optional[float],
    th_accept: float,
    th_reject: float,
    th_final: float,
) -> Tuple[str, float]:
    if p_good_1 >= th_accept:
        return "ACCEPT", p_good_1
    if p_good_1 <= th_reject:
        return "REJECT", p_good_1

    if p_good_2 is None:
        return "UNSURE", p_good_1

    avg = 0.5 * (p_good_1 + p_good_2)
    return ("ACCEPT" if avg >= th_final else "REJECT"), avg


def run_stageB_and_chute(
    model_b: tf.keras.Model,
    class_names_b: list[str],
    x: np.ndarray,
) -> Tuple[int, str, float]:
    """Run Stage B on preprocessed input. Returns (chute_id, label, confidence)."""
    x_b = match_model_channels(x, model_b)
    out = model_b.predict(x_b, verbose=0)
    probs = np.asarray(out)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    label = class_names_b[idx]
    # Chute 0 = good (Stage A). Stage B faults -> chutes 1, 2, 3, ...
    chute_id = 1 + idx
    return chute_id, label, conf


class BeltController:
    """Hardware-safe placeholder controller.

    - forward(): advance belt (e.g. part moves to next position).
    - reverse(seconds): briefly reverse (for recheck).
    - stop(): halt belt (legacy; use divert + forward for chutes).
    - divert(chute_id): route part to chute. 0 = good, 1+ = reject/fault-type.
      Replace with GPIO/serial when wiring to real hardware.
    """

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run

    def forward(self) -> None:
        if self.dry_run:
            print("[BELT] FORWARD")

    def reverse(self, seconds: float) -> None:
        if self.dry_run:
            print(f"[BELT] REVERSE for {seconds:.2f}s")
        time.sleep(seconds)

    def stop(self) -> None:
        if self.dry_run:
            print("[BELT] STOP")

    def divert(self, chute_id: int) -> None:
        """Route part to chute. 0=good, 1=reject (or fault-type 1), 2–4=fault types."""
        if self.dry_run:
            print(f"[CHUTE] DIVERT -> {chute_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-time cap classifier with optional UNSURE recheck.")
    parser.add_argument("--model", type=str, default="models/cap_classifier_best.keras")
    parser.add_argument("--class_names", type=str, default="models/class_names.json")
    parser.add_argument("--stageB_model", type=str, default=None, help="Stage B fault-type model (optional)")
    parser.add_argument("--stageB_classes", type=str, default=None, help="Stage B class_names JSON (optional)")

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    parser.add_argument("--th_accept", type=float, default=0.60, help="ACCEPT if p_good >= this")
    parser.add_argument("--th_reject", type=float, default=0.40, help="REJECT if p_good <= this")
    parser.add_argument("--th_final", type=float, default=0.60, help="After recheck, accept if avg >= this")

    parser.add_argument("--reverse_seconds", type=float, default=0.80)
    parser.add_argument("--no_recheck", action="store_true", help="Disable the second-sample recheck")
    parser.add_argument(
        "--no_dry_run",
        action="store_true",
        dest="no_dry_run",
        help="Actually control hardware (default: dry run, print [BELT]/[CHUTE] only)",
    )

    # ROI as fractions of frame width/height
    parser.add_argument("--roi", type=float, nargs=4, default=[0.30, 0.70, 0.15, 0.55],
                        metavar=("X1","X2","Y1","Y2"),
                        help="ROI crop as fractions: x1 x2 y1 y2")
    parser.add_argument("--show_roi", action="store_true", help="Draw ROI box on preview")

    args = parser.parse_args()
    dry_run = not getattr(args, "no_dry_run", False)

    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names(args.class_names)

    if "good" not in class_names or "faulty" not in class_names:
        raise SystemExit(f"class_names must include 'good' and 'faulty'. Got: {class_names}")

    use_stageB = bool(args.stageB_model and args.stageB_classes)
    model_b: Optional[tf.keras.Model] = None
    class_names_b: Optional[list[str]] = None
    if use_stageB:
        model_b = tf.keras.models.load_model(args.stageB_model)
        class_names_b = load_class_names(args.stageB_classes)
        print(f"Two-stage mode: Stage B classes = {class_names_b}")

    image_size = tuple(int(x) for x in model.input_shape[1:3])  # type: ignore

    belt = BeltController(dry_run=dry_run)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise SystemExit("Could not open camera.")

    mode_str = "two-stage (multi-chute)" if use_stageB else "binary (accept/reject)"
    print(f"\nMode: {mode_str}")
    print("Controls:\n  SPACE = classify current frame\n  Q     = quit\n")

    last_text = "READY"
    last_p: Optional[float] = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        overlay = frame.copy()

        if args.show_roi:
            h, w = frame.shape[:2]
            rx1, rx2, ry1, ry2 = args.roi
            x1 = int(w * rx1); x2 = int(w * rx2)
            y1 = int(h * ry1); y2 = int(h * ry2)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)

        msg = last_text if last_p is None else f"{last_text} | p_good={last_p:.3f}"
        cv2.putText(overlay, msg, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(overlay, "SPACE=classify | Q=quit", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Bottle Cap Sorter (recheck)", overlay)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            break

        if key == 32:  # SPACE
            x1 = preprocess_frame(frame, image_size=image_size, roi=tuple(args.roi))
            x1 = match_model_channels(x1, model)
            p1 = _infer_p_good_from_output(model.predict(x1, verbose=0), class_names)

            decision1, score1 = decide_with_recheck(
                p_good_1=p1,
                p_good_2=None,
                th_accept=args.th_accept,
                th_reject=args.th_reject,
                th_final=args.th_final,
            )

            if decision1 != "UNSURE" or args.no_recheck:
                last_text = decision1
                last_p = score1
                print(f"[PASS1] p_good={p1:.3f} -> {decision1}")
                x_last = x1
                if decision1 == "ACCEPT":
                    belt.divert(0)
                    belt.forward()
                else:
                    if use_stageB and model_b is not None and class_names_b is not None:
                        chute_id, fault_label, fault_conf = run_stageB_and_chute(model_b, class_names_b, x_last)
                        print(f"  StageB: {fault_label} (conf={fault_conf:.3f}) -> chute {chute_id}")
                        belt.divert(chute_id)
                    else:
                        belt.divert(1)
                    belt.forward()
                continue

            # Recheck flow
            print(f"[PASS1] p_good={p1:.3f} -> UNSURE -> reversing for recheck...")
            belt.reverse(args.reverse_seconds)

            ok2, frame2 = cap.read()
            if not ok2:
                last_text = "REJECT"
                last_p = p1
                print("[PASS2] Could not read frame -> REJECT (safe)")
                belt.divert(1)
                belt.forward()
                continue

            x2 = preprocess_frame(frame2, image_size=image_size, roi=tuple(args.roi))
            x2 = match_model_channels(x2, model)
            p2 = _infer_p_good_from_output(model.predict(x2, verbose=0), class_names)

            final_decision, final_score = decide_with_recheck(
                p_good_1=p1,
                p_good_2=p2,
                th_accept=args.th_accept,
                th_reject=args.th_reject,
                th_final=args.th_final,
            )

            last_text = final_decision + " (RECHECK)"
            last_p = final_score
            print(f"[PASS2] p_good2={p2:.3f} -> FINAL avg={final_score:.3f} -> {final_decision}")

            x_last = x2
            if final_decision == "ACCEPT":
                belt.divert(0)
                belt.forward()
            else:
                if use_stageB and model_b is not None and class_names_b is not None:
                    chute_id, fault_label, fault_conf = run_stageB_and_chute(model_b, class_names_b, x_last)
                    print(f"  StageB: {fault_label} (conf={fault_conf:.3f}) -> chute {chute_id}")
                    belt.divert(chute_id)
                else:
                    belt.divert(1)
                belt.forward()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
