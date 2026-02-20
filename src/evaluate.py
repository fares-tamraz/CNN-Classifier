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

    from data_loader import load_datasets

    image_size = (int(args.image_size[0]), int(args.image_size[1]))

    # Load model with custom Lambda layer support
    # Keras >= 3.0 has issues deserializing Lambda layers, so we handle it gracefully
    try:
        model = tf.keras.models.load_model(args.model_path, compile=False)
    except (NotImplementedError, ValueError) as e:
        # If deserialization fails due to Lambda layer, try with safe_mode=False
        print(f"Note: Standard load failed ({type(e).__name__}), attempting workaround...")
        model = tf.keras.models.load_model(
            args.model_path, 
            safe_mode=False, 
            compile=False
        )
    
    class_names = load_class_names(args.class_names)

    _, _, test_ds, ds_class_names = load_datasets(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        color_mode=args.color_mode,
        augment=False,
        seed=42,
        verify_class_order=True,
        verbose=False,
    )

    if list(ds_class_names) != list(class_names):
        print("Warning: class_names.json differs from dataset class order.")
        print("  class_names.json:", class_names)
        print("  dataset order:   ", ds_class_names)
        class_names = list(ds_class_names)

    # Predict
    y_true = []
    y_pred = []
    y_probs = []

    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        p = np.array(p)
        
        # Handle different output formats
        if p.ndim == 2:
            if p.shape[1] == 1:
                # Binary sigmoid output
                y_probs.extend(p[:, 0].tolist())
                pred_class = (p[:, 0] >= args.threshold).astype(int)
            else:
                # Multi-class softmax output
                y_probs.append(p)
                pred_class = np.argmax(p, axis=1)
            
            y_pred.extend(pred_class.tolist())
        else:
            raise ValueError(f"Unexpected model output shape: {p.shape}")

        y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    
    # Check if binary or multi-class
    num_classes = len(class_names)
    
    if num_classes == 2:
        # Binary classification - original logic
        if set(class_names) != {"faulty", "good"}:
            raise SystemExit(f"Expected binary classes {{faulty,good}}. Got: {class_names}")
        
        y_prob = np.array(y_probs, dtype=float)
        good_idx = class_names.index("good")
        
        if good_idx == 1:
            y_true_good = y_true
        else:
            y_true_good = 1 - y_true
        
        y_pred_good = (y_prob >= args.threshold).astype(int)
        
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
    
    else:
        # Multi-class classification
        from sklearn.metrics import classification_report, confusion_matrix
        
        acc = np.mean(y_true == y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
        print("\n=== Test results ===")
        print(f"Classes: {class_names}")
        print(f"Overall accuracy = {acc:.4f}\n")
        
        print("Confusion matrix:")
        header = "".join(f"{c[:8]:>10}" for c in class_names)
        print(f"{'':15}{header}")
        print("-" * (15 + 10 * num_classes))
        for i, row in enumerate(cm):
            row_str = "".join(f"{v:>10}" for v in row)
            print(f"{class_names[i][:15]:15}{row_str}")
        
        print("\nPer-class metrics:")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))


if __name__ == "__main__":
    main()
