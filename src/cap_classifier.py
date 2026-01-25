"""Cap classifier API for conveyor integration.

Your (ML/OpenCV) deliverable: **image in → good/faulty + chute_id out**.
Electrical calls this when they have a frame; you don't handle camera, trigger, or servos.

Usage:
  from src.cap_classifier import CapClassifier, classify_cap

  clf = CapClassifier("models/cap_classifier_best.keras", "models/class_names.json")
  result = clf.classify(frame)   # frame: BGR ndarray or path
  # result["decision"] "good"|"faulty"|"unsure", result["chute_id"], ...
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf


def load_class_names(path: Union[str, Path]) -> List[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "class_names" in data:
        return [str(x) for x in data["class_names"]]
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


def _preprocess(
    frame_bgr: np.ndarray,
    image_size: Tuple[int, int],
    roi: Tuple[float, float, float, float],
) -> np.ndarray:
    """Crop ROI, grayscale, resize, normalize. Returns (1,H,W,1) float32."""
    h, w = frame_bgr.shape[:2]
    rx1, rx2, ry1, ry2 = roi
    x1 = max(0, min(int(w * rx1), w - 1))
    x2 = max(x1 + 1, min(int(w * rx2), w))
    y1 = max(0, min(int(h * ry1), h - 1))
    y2 = max(y1 + 1, min(int(h * ry2), h))
    crop = frame_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.resize(gray, image_size, interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32) / 255.0
    return np.expand_dims(x, axis=(0, -1))


def _match_channels(x: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    shp = model.input_shape
    if isinstance(shp, list):
        shp = shp[0]
    ch = shp[-1]
    if ch == 1:
        return x
    if ch == 3:
        return np.repeat(x, repeats=3, axis=-1)
    raise ValueError(f"Unsupported model input channels: {ch}")


def _infer_p_good(out: np.ndarray, class_names: List[str]) -> float:
    out = np.asarray(out)
    if out.ndim == 2 and out.shape[1] == 1:
        p = float(out[0, 0])
        if len(class_names) == 2 and class_names[1] == "good":
            return p
        if len(class_names) == 2 and class_names[1] == "faulty":
            return 1.0 - p
        return p
    if out.ndim == 2 and out.shape[1] == 2:
        probs = out[0]
        if "good" in class_names:
            return float(probs[class_names.index("good")])
        return float(probs[1])
    raise ValueError(f"Unexpected model output shape: {out.shape}")


def _decision(p_good: float, th_accept: float, th_reject: float) -> str:
    if p_good >= th_accept:
        return "good"
    if p_good <= th_reject:
        return "faulty"
    return "unsure"


class CapClassifier:
    """Load-once, classify-many API for electrical integration.

    Electrical loads this at startup, then calls classify(frame) each time
    they have a new cap image (e.g. after trigger → pause → capture).
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        class_names_path: Union[str, Path],
        *,
        stageB_model_path: Optional[Union[str, Path]] = None,
        stageB_classes_path: Optional[Union[str, Path]] = None,
        th_accept: float = 0.6,
        th_reject: float = 0.4,
        roi: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    ):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.th_accept = th_accept
        self.th_reject = th_reject
        self.roi = roi

        self.model = tf.keras.models.load_model(str(self.model_path))
        self.class_names = load_class_names(self.class_names_path)
        if set(self.class_names) != {"faulty", "good"}:
            raise ValueError(f"Stage A must have class_names ['faulty','good']. Got: {self.class_names}")

        shp = self.model.input_shape
        if isinstance(shp, list):
            shp = shp[0]
        self.image_size = (int(shp[1]), int(shp[2]))
        self.model_b: Optional[tf.keras.Model] = None
        self.class_names_b: Optional[List[str]] = None
        if stageB_model_path and stageB_classes_path:
            self.model_b = tf.keras.models.load_model(str(stageB_model_path))
            self.class_names_b = load_class_names(stageB_classes_path)

    def classify(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> Dict[str, Any]:
        """Run inference on one cap image.

        Args:
            image: File path to image, or BGR numpy array (H,W,3) from OpenCV.

        Returns:
            {
                "decision": "good" | "faulty" | "unsure",
                "p_good": float,
                "chute_id": int,       # 0=good, 1+=reject/fault-type
                "fault_type": str | None,
                "fault_confidence": float | None,
            }
        """
        if isinstance(image, (str, Path)):
            frame = cv2.imread(str(image))
            if frame is None:
                raise FileNotFoundError(f"Could not load image: {image}")
        else:
            frame = np.asarray(image)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("image must be BGR (H,W,3) or a path")

        x = _preprocess(frame, self.image_size, self.roi)
        x = _match_channels(x, self.model)
        out = self.model.predict(x, verbose=0)
        p_good = _infer_p_good(out, self.class_names)
        decision = _decision(p_good, self.th_accept, self.th_reject)

        chute_id = 0 if decision == "good" else 1
        fault_type: Optional[str] = None
        fault_confidence: Optional[float] = None

        if decision == "faulty" and self.model_b is not None and self.class_names_b is not None:
            x_b = _match_channels(x, self.model_b)
            pred_b = self.model_b.predict(x_b, verbose=0)
            probs = np.asarray(pred_b)[0]
            idx = int(np.argmax(probs))
            fault_type = self.class_names_b[idx]
            fault_confidence = float(probs[idx])
            chute_id = 1 + idx

        return {
            "decision": decision,
            "p_good": p_good,
            "chute_id": chute_id,
            "fault_type": fault_type,
            "fault_confidence": fault_confidence,
        }


def classify_cap(
    image: Union[str, Path, np.ndarray],
    model_path: Union[str, Path],
    class_names_path: Union[str, Path],
    *,
    stageB_model_path: Optional[Union[str, Path]] = None,
    stageB_classes_path: Optional[Union[str, Path]] = None,
    th_accept: float = 0.6,
    th_reject: float = 0.4,
    roi: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> Dict[str, Any]:
    """One-shot classify. For loops, use CapClassifier and call .classify() repeatedly."""
    clf = CapClassifier(
        model_path,
        class_names_path,
        stageB_model_path=stageB_model_path,
        stageB_classes_path=stageB_classes_path,
        th_accept=th_accept,
        th_reject=th_reject,
        roi=roi,
    )
    return clf.classify(image)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Classify one cap image (CLI for scripts).")
    ap.add_argument("--model", default="models/cap_classifier_best.keras")
    ap.add_argument("--class_names", default="models/class_names.json")
    ap.add_argument("--stageB_model", default=None)
    ap.add_argument("--stageB_classes", default=None)
    ap.add_argument("--th_accept", type=float, default=0.6)
    ap.add_argument("--th_reject", type=float, default=0.4)
    ap.add_argument("--roi", type=float, nargs=4, default=[0.0, 1.0, 0.0, 1.0], metavar=("X1", "X2", "Y1", "Y2"))
    ap.add_argument("image", help="Path to image file.")
    args = ap.parse_args()

    out = classify_cap(
        args.image,
        args.model,
        args.class_names,
        stageB_model_path=args.stageB_model,
        stageB_classes_path=args.stageB_classes,
        th_accept=args.th_accept,
        th_reject=args.th_reject,
        roi=tuple(args.roi),
    )
    print(json.dumps(out, indent=2))
