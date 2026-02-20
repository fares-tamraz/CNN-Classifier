#!/usr/bin/env python3
"""Full evaluation of the v3 model with detailed metrics and confusion matrix."""

import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from src.data_loader import load_datasets


def main():
    # Load model - use v7 by default, fallback to v6, v5, v4, v3
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "models/v7/cap_classifier_best.keras"
        if not Path(model_path).exists():
            model_path = "models/v6/cap_classifier_best.keras"
        if not Path(model_path).exists():
            model_path = "models/v5/cap_classifier_best.keras"
        if not Path(model_path).exists():
            model_path = "models/v4/cap_classifier_best.keras"
        if not Path(model_path).exists():
            model_path = "models/v3/cap_classifier_best.keras"
    
    # Try to find class_names.json near the model
    model_dir = Path(model_path).parent
    class_names_path = model_dir / "class_names.json"
    if not class_names_path.exists():
        class_names_path = Path("models/class_names.json")
    if not class_names_path.exists():
        class_names_path = Path("models/v6/class_names.json")
    
    # Auto-detect dataset based on model
    if any(v in str(model_path) for v in ("v7", "v6", "v5")):
        data_dir = "data/processed/cls_3class_crops"
    else:
        data_dir = "data/processed/cls_5class_crops_v3"
    
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    class_names = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
    print(f"Classes: {class_names}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    _, _, test_ds, ds_class_names = load_datasets(
        data_dir,
        image_size=(224, 224),
        batch_size=32,
        color_mode="grayscale",
        augment=False,
        seed=42,
        verify_class_order=True,
        verbose=True,
    )
    
    # Get predictions
    print("\nRunning predictions on full test set...")
    y_true = []
    y_pred = []
    y_probs = []
    
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        pred_class = np.argmax(p, axis=1)
        y_pred.extend(pred_class.tolist())
        y_probs.append(p)
        y_true.extend(y.numpy().tolist())
    
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_probs = np.vstack(y_probs)
    
    # Overall accuracy
    acc = np.mean(y_true == y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    print("\n" + "="*80)
    print("=== FULL TEST SET EVALUATION ===")
    print("="*80)
    print(f"\nTotal test samples: {len(y_true)}")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    
    # Class distribution
    print("True class distribution:")
    for i, name in enumerate(class_names):
        count = np.sum(y_true == i)
        print(f"  {name:15s}: {count:4d} samples")
    
    print("\n" + "="*80)
    print("CONFUSION MATRIX")
    print("="*80)
    print("(rows=true class, cols=predicted class)\n")
    
    # Header
    header = "".join(f"{c[:8]:>10}" for c in class_names)
    print(f"{'':15}{header}")
    print("-" * (15 + 10 * len(class_names)))
    
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>10}" for v in row)
        print(f"{class_names[i][:15]:15}{row_str}")
    
    print("\n" + "="*80)
    print("PER-CLASS METRICS")
    print("="*80 + "\n")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # Analyze good vs faulty (if good class exists)
    good_class_name = None
    if 'good_cap' in class_names:
        good_class_name = 'good_cap'
    elif 'good' in class_names:
        good_class_name = 'good'
    
    if good_class_name:
        print("\n" + "="*80)
        print("GOOD vs FAULTY ANALYSIS")
        print("="*80 + "\n")
        
        good_idx = class_names.index(good_class_name)
        y_true_binary = (y_true == good_idx).astype(int)
        y_pred_binary = (y_pred == good_idx).astype(int)
        
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
        
        binary_acc = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Binary classification (good vs faulty):")
        print(f"  Accuracy:  {binary_acc:.4f}")
        print(f"  Precision: {precision:.4f} (of predicted good, how many are actually good)")
        print(f"  Recall:    {recall:.4f} (of actual good, how many we caught)")
        print(f"  F1-score:  {f1:.4f}")
        print(f"\nConfusion (good=positive):")
        print(f"  TP={tp:4d}  FP={fp:4d}")
        print(f"  FN={fn:4d}  TN={tn:4d}")
        
        # Analyze p_good distribution
        print("\n" + "="*80)
        print("P_GOOD DISTRIBUTION ANALYSIS")
        print("="*80 + "\n")
        
        p_good = y_probs[:, good_idx]
        
        for i, name in enumerate(class_names):
            mask = y_true == i
            if np.sum(mask) == 0:
                continue
            p_good_class = p_good[mask]
            print(f"{name:15s}: mean={np.mean(p_good_class):.4f}, "
                  f"std={np.std(p_good_class):.4f}, "
                  f"min={np.min(p_good_class):.4f}, "
                  f"max={np.max(p_good_class):.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
