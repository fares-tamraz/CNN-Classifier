#!/usr/bin/env python3
"""Test model on actual dataset images to verify it works properly."""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path

# FocalLoss from train.py
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_true_onehot = tf.one_hot(tf.cast(y_true, tf.int32), 
                                   tf.shape(y_pred)[-1])
        y_true_onehot = tf.cast(y_true_onehot, y_pred.dtype)
        
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce = -y_true_onehot * tf.math.log(y_pred)
        ce = tf.reduce_sum(ce, axis=-1)
        
        p_t = tf.reduce_sum(y_true_onehot * y_pred, axis=-1)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        focal_loss = self.alpha * focal_weight * ce
        
        return tf.reduce_mean(focal_loss)


def load_class_names(path: str) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict) and "class_names" in data:
        return [str(x) for x in data["class_names"]]
    raise ValueError(f"Unexpected format in {path}: {type(data)}")


def preprocess_image(img_path: str, image_size: tuple = (224, 224)) -> np.ndarray:
    """Load image, convert to grayscale, normalize.
    
    IMPORTANT: Must match training preprocessing (no GaussianBlur).
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    # Removed GaussianBlur to match training preprocessing
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1,H,W,1)
    return img


def _infer_p_good_from_output(y: np.ndarray, class_names: list[str]) -> float:
    """Extract p_good from model output."""
    y = np.asarray(y)
    if y.ndim == 2 and y.shape[1] >= 2:
        probs = y[0]
        for good_name in ['good_cap', 'good']:
            if good_name in class_names:
                return float(probs[class_names.index(good_name)])
        return float(probs[-1])
    raise ValueError(f"Unexpected shape: {y.shape}")


def test_model(model_path: str, class_names_path: str, test_dir: str):
    """Test model on images in a directory."""
    # Load model
    model = tf.keras.models.load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
    class_names = load_class_names(class_names_path)
    image_size = tuple(int(x) for x in model.input_shape[1:3])
    
    print(f"Model: {model_path}")
    print(f"Classes: {class_names}")
    print(f"Image size: {image_size}")
    print(f"Testing directory: {test_dir}\n")
    
    # Get all images
    import glob
    images = glob.glob(os.path.join(test_dir, "*.png")) + glob.glob(os.path.join(test_dir, "*.jpg"))
    images = sorted(images)[:20]  # Test first 20
    
    print(f"Testing {len(images)} images...\n")
    
    confidences = []
    for img_path in images:
        x = preprocess_image(img_path, image_size)
        if x is None:
            print(f"  SKIP: {os.path.basename(img_path)} (failed to load)")
            continue
        
        # Match channels
        ch = model.input_shape[-1]
        if ch == 3 and x.shape[-1] == 1:
            x = np.repeat(x, repeats=3, axis=-1)
        
        out = model.predict(x, verbose=0)
        p_good = _infer_p_good_from_output(out, class_names)
        confidences.append(p_good)
        
        print(f"  {os.path.basename(img_path)}: p_good={p_good:.4f}")
    
    if confidences:
        print(f"\nStats:")
        print(f"  Mean: {np.mean(confidences):.4f}")
        print(f"  Std:  {np.std(confidences):.4f}")
        print(f"  Min:  {np.min(confidences):.4f}")
        print(f"  Max:  {np.max(confidences):.4f}")


if __name__ == "__main__":
    # Test v3 model on good caps
    print("=" * 60)
    print("TESTING V3 MODEL ON GOOD CAPS FROM DATASET")
    print("=" * 60 + "\n")
    
    test_model(
        "models/v3/cap_classifier_best.keras",
        "models/class_names.json",
        "data/processed/cls_5class_crops_v3/test/good_cap"
    )
    
    print("\n" + "=" * 60)
    print("TESTING V3 MODEL ON BROKEN CAPS FROM DATASET")
    print("=" * 60 + "\n")
    
    test_model(
        "models/v3/cap_classifier_best.keras",
        "models/class_names.json",
        "data/processed/cls_5class_crops_v3/test/broken_cap"
    )
