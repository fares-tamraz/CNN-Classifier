#!/usr/bin/env python3
"""Audit dataset labels using trained model to find mislabeled images.

Uses batch predictions for speed. Flags images where model strongly disagrees with label.
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import tensorflow as tf


def preprocess_image(img_path: str, image_size: tuple = (224, 224)) -> np.ndarray:
    """Load and preprocess single image."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


def collect_all_images(data_dir: Path, class_names: list) -> list:
    """Collect all image paths with their labels."""
    images = []
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            true_label = class_dir.name
            if true_label not in class_names:
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                images.append({
                    "path": str(img_path),
                    "split": split,
                    "true_label": true_label,
                    "true_idx": class_names.index(true_label)
                })
    return images


def audit_dataset_batch(
    data_dir: str,
    model_path: str,
    class_names: list,
    confidence_threshold: float = 0.6,
    batch_size: int = 32
) -> dict:
    """Audit all images using batch predictions."""
    
    print("Loading model...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (NotImplementedError, ValueError):
        model = tf.keras.models.load_model(model_path, safe_mode=False, compile=False)
    image_size = tuple(int(x) for x in model.input_shape[1:3])
    n_channels = model.input_shape[-1]
    print(f"Model input: {image_size}, channels: {n_channels}")
    
    data_dir = Path(data_dir)
    
    print("Collecting images...")
    all_images = collect_all_images(data_dir, class_names)
    print(f"Found {len(all_images)} images")
    
    # Prepare batches and predict
    flagged = defaultdict(list)
    stats = {
        "total": len(all_images),
        "flagged": 0,
        "per_class": defaultdict(lambda: {"total": 0, "flagged": 0})
    }
    
    # Process in batches
    n_batches = (len(all_images) + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(all_images))
        batch_items = all_images[start:end]
        
        print(f"\rProcessing batch {batch_idx+1}/{n_batches} ({start}-{end})...", end="", flush=True)
        
        # Load batch images
        batch_images = []
        valid_items = []
        
        for item in batch_items:
            img = preprocess_image(item["path"], image_size)
            if img is not None:
                # Add channel dimension
                img = np.expand_dims(img, axis=-1)
                # Match channels if needed
                if n_channels == 3:
                    img = np.repeat(img, 3, axis=-1)
                batch_images.append(img)
                valid_items.append(item)
        
        if not batch_images:
            continue
        
        # Batch predict
        batch_x = np.array(batch_images)
        batch_probs = model.predict(batch_x, verbose=0)
        
        # Process results
        for item, probs in zip(valid_items, batch_probs):
            true_label = item["true_label"]
            true_idx = item["true_idx"]
            
            stats["per_class"][true_label]["total"] += 1
            
            pred_idx = np.argmax(probs)
            pred_label = class_names[pred_idx]
            pred_conf = probs[pred_idx]
            true_conf = probs[true_idx]
            
            # Flag if model confidently predicts different class
            if pred_idx != true_idx and pred_conf >= confidence_threshold:
                stats["flagged"] += 1
                stats["per_class"][true_label]["flagged"] += 1
                
                flagged[(true_label, pred_label)].append({
                    "path": item["path"],
                    "split": item["split"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_conf": float(pred_conf),
                    "true_conf": float(true_conf),
                })
    
    print("\nDone!")
    # Convert tuple keys to string for JSON serialization
    flagged_serializable = {f"{k[0]}__to__{k[1]}": v for k, v in flagged.items()}
    return {"flagged": flagged_serializable, "stats": dict(stats)}


def print_report(results: dict, class_names: list):
    """Print formatted audit report."""
    stats = results["stats"]
    # Convert back from serialized keys
    flagged = {}
    for k, v in results["flagged"].items():
        parts = k.split("__to__")
        flagged[(parts[0], parts[1])] = v
    
    print("\n" + "="*80)
    print("LABEL AUDIT REPORT")
    print("="*80)
    
    total = stats["total"]
    total_flagged = stats["flagged"]
    print(f"\nTotal images scanned: {total}")
    print(f"Flagged as potential mislabels: {total_flagged} ({100*total_flagged/max(1,total):.1f}%)")
    
    print("\n" + "-"*60)
    print("Per-class breakdown:")
    print("-"*60)
    
    for class_name in class_names:
        cs = stats["per_class"].get(class_name, {"total": 0, "flagged": 0})
        if cs["total"] > 0:
            pct = 100 * cs["flagged"] / cs["total"]
            print(f"  {class_name:15s}: {cs['flagged']:4d} / {cs['total']:4d} flagged ({pct:5.1f}%)")
    
    print("\n" + "-"*60)
    print("Mislabel patterns (true_label -> predicted_label):")
    print("-"*60)
    
    sorted_patterns = sorted(flagged.items(), key=lambda x: -len(x[1]))
    
    for (true_label, pred_label), items in sorted_patterns:
        print(f"\n  {true_label} -> {pred_label}: {len(items)} images")
        for item in items[:3]:
            path = Path(item["path"])
            print(f"    - {path.parent.parent.name}/{path.name} (conf={item['pred_conf']:.2f})")
        if len(items) > 3:
            print(f"    ... and {len(items)-3} more")
    
    print("\n" + "="*80)


def copy_flagged_images(results: dict, review_dir: str):
    """Copy flagged images to review folder."""
    review_dir = Path(review_dir)
    # Convert back from serialized keys
    flagged = {}
    for k, v in results["flagged"].items():
        parts = k.split("__to__")
        flagged[(parts[0], parts[1])] = v
    
    moved = 0
    for (true_label, pred_label), items in flagged.items():
        pattern_dir = review_dir / f"{true_label}_to_{pred_label}"
        pattern_dir.mkdir(parents=True, exist_ok=True)
        
        for item in items:
            src = Path(item["path"])
            dst = pattern_dir / src.name
            if src.exists():
                shutil.copy2(src, dst)
                moved += 1
    
    print(f"\nCopied {moved} flagged images to: {review_dir}")


def main():
    parser = argparse.ArgumentParser(description="Audit dataset labels")
    parser.add_argument("--data_dir", default="data/processed/cls_5class_crops_v3")
    parser.add_argument("--model", default="models/v4/cap_classifier_best.keras")
    parser.add_argument("--class_names", default="models/v4/class_names.json")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--copy_flagged", action="store_true")
    parser.add_argument("--review_dir", default="data/review_mislabels")
    parser.add_argument("--output", default="data/label_audit_results.json")
    args = parser.parse_args()
    
    # Load class names
    class_names_path = Path(args.class_names)
    if not class_names_path.exists():
        class_names_path = Path("models/class_names.json")
    class_names = json.loads(class_names_path.read_text(encoding="utf-8"))
    
    print(f"Auditing: {args.data_dir}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Classes: {class_names}")
    
    results = audit_dataset_batch(
        args.data_dir,
        args.model,
        class_names,
        confidence_threshold=args.threshold,
        batch_size=args.batch_size
    )
    
    print_report(results, class_names)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to: {output_path}")
    
    if args.copy_flagged:
        copy_flagged_images(results, args.review_dir)


if __name__ == "__main__":
    main()
