#!/usr/bin/env python3
"""Dataset evaluation script - bottle cap inspection project.

Covers:
  1. Image count per split and per class (flags severe imbalance)
  2. Image integrity: corrupt files, 0-byte files, unreadable images
  3. Class distribution bar chart (saved to <out_dir>/class_distribution.png)
  4. Augmentation preview  (saved to <out_dir>/augmentation_preview.png)
  5. Optional model evaluation (requires --model):
       - Confusion matrix on test set
       - Precision / Recall / F1 per class
       - *** FAULTY recall prominently highlighted ***

Usage:
  # Dataset integrity only (no TensorFlow required):
  python evaluate_dataset.py --data_dir data/processed/cls_3class_crops

  # Full eval with model:
  python evaluate_dataset.py \\
    --data_dir data/processed/cls_3class_crops \\
    --model models/v6/cap_classifier_best.keras \\
    --class_names models/v6/class_names.json

  # Also check raw YOLO annotation integrity:
  python evaluate_dataset.py \\
    --data_dir data/processed/cls_3class_crops \\
    --yolo_dir data/raw/bottle-cap-yolo \\
    --model models/v6/cap_classifier_best.keras \\
    --class_names models/v6/class_names.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# PIL is a soft requirement - only needed for augmentation preview
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SPLITS = ("train", "val", "test")
SEP = "=" * 70


# =============================================================================
# SECTION 1: Class distribution
# =============================================================================

def count_images_per_class(data_dir: Path) -> dict[str, dict[str, int]]:
    """Return {split: {class_name: count}} for all found splits."""
    result: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        class_counts: dict[str, int] = {}
        for cls_dir in sorted(split_dir.iterdir()):
            if cls_dir.is_dir():
                images = [
                    f for f in cls_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                ]
                class_counts[cls_dir.name] = len(images)
        result[split] = class_counts
    return result


def print_distribution(counts: dict[str, dict[str, int]]) -> None:
    print(SEP)
    print("CLASS DISTRIBUTION")
    print(SEP)

    all_classes: set[str] = set()
    for split_counts in counts.values():
        all_classes.update(split_counts.keys())
    classes = sorted(all_classes)

    # Header
    col_w = 12
    header = f"{'Class':<20}" + "".join(f"{s:>{col_w}}" for s in counts.keys())
    print(header)
    print("-" * len(header))

    # Rows
    for cls in classes:
        row = f"{cls:<20}" + "".join(
            f"{counts[s].get(cls, 0):>{col_w}}" for s in counts.keys()
        )
        print(row)

    # Totals
    totals = {s: sum(v.values()) for s, v in counts.items()}
    print("-" * len(header))
    total_row = f"{'TOTAL':<20}" + "".join(f"{totals.get(s, 0):>{col_w}}" for s in counts.keys())
    print(total_row)

    # Imbalance warnings per split
    print()
    faulty_variants = {"faulty", "broken_cap", "broken_ring", "loose_cap"}
    for split, split_counts in counts.items():
        if not split_counts:
            continue
        total = sum(split_counts.values())
        good = split_counts.get("good", split_counts.get("good_cap", 0))
        faulty = sum(split_counts.get(k, 0) for k in faulty_variants if k in split_counts)
        if total > 0 and faulty > 0 and good > 0:
            ratio = good / faulty
            flag = " [!] IMBALANCED" if ratio > 3 or ratio < 0.33 else ""
            print(f"  [{split}] good:faulty ratio = {ratio:.2f}{flag}")


def save_distribution_chart(counts: dict[str, dict[str, int]], out_dir: Path) -> None:
    """Save a class distribution bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed - skipping distribution chart")
        return

    all_classes = sorted({c for sc in counts.values() for c in sc})
    x = np.arange(len(all_classes))
    width = 0.25
    splits = list(counts.keys())

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, split in enumerate(splits):
        vals = [counts[split].get(c, 0) for c in all_classes]
        ax.bar(x + i * width, vals, width, label=split)

    ax.set_title("Class Distribution by Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image Count")
    ax.set_xticks(x + width)
    ax.set_xticklabels(all_classes, rotation=15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out_path = out_dir / "class_distribution.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# SECTION 2: Image integrity
# =============================================================================

def check_image_integrity(data_dir: Path) -> dict:
    """Check for corrupt, 0-byte, and unreadable images."""
    corrupt: list[str] = []
    zero_byte: list[str] = []
    ok = 0
    total = 0

    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        for cls_dir in split_dir.iterdir():
            if not cls_dir.is_dir():
                continue
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    continue
                total += 1
                size = img_path.stat().st_size
                if size == 0:
                    zero_byte.append(str(img_path.relative_to(data_dir)))
                    continue
                if HAS_PIL:
                    try:
                        with PILImage.open(img_path) as im:
                            im.verify()
                        ok += 1
                    except Exception:
                        corrupt.append(str(img_path.relative_to(data_dir)))
                else:
                    ok += 1

    return {"total": total, "ok": ok, "corrupt": corrupt, "zero_byte": zero_byte}


def print_integrity_report(report: dict) -> None:
    print(SEP)
    print("IMAGE INTEGRITY")
    print(SEP)
    print(f"  Total images scanned : {report['total']}")
    print(f"  OK                   : {report['ok']}")
    print(f"  Zero-byte files      : {len(report['zero_byte'])}")
    print(f"  Corrupt / unreadable : {len(report['corrupt'])}")

    if report["zero_byte"]:
        print("\n  Zero-byte files:")
        for p in report["zero_byte"][:10]:
            print(f"    {p}")
        if len(report["zero_byte"]) > 10:
            print(f"    ... and {len(report['zero_byte']) - 10} more")

    if report["corrupt"]:
        print("\n  Corrupt files:")
        for p in report["corrupt"][:10]:
            print(f"    {p}")
        if len(report["corrupt"]) > 10:
            print(f"    ... and {len(report['corrupt']) - 10} more")

    if not report["zero_byte"] and not report["corrupt"]:
        print("  All images are readable. OK")


# =============================================================================
# SECTION 3: YOLO annotation integrity (optional)
# =============================================================================

def check_yolo_annotations(yolo_dir: Path) -> None:
    print(SEP)
    print(f"YOLO ANNOTATION INTEGRITY: {yolo_dir.name}")
    print(SEP)

    yolo_splits = ["train", "valid", "val", "test"]
    issues: list[str] = []

    for split in yolo_splits:
        img_dir = yolo_dir / split / "images"
        lbl_dir = yolo_dir / split / "labels"

        if not img_dir.is_dir():
            continue

        img_files = {f.stem for f in img_dir.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
        lbl_files = {f.stem for f in lbl_dir.iterdir()
                     if f.suffix == ".txt"} if lbl_dir.is_dir() else set()

        orphan_imgs = img_files - lbl_files   # images with no label
        orphan_lbls = lbl_files - img_files   # labels with no image

        print(f"\n  [{split}] images={len(img_files)}  labels={len(lbl_files)}")
        if orphan_imgs:
            print(f"    [!]  {len(orphan_imgs)} image(s) with no label file")
            for s in list(orphan_imgs)[:3]:
                print(f"       {s}")
        if orphan_lbls:
            print(f"    [!]  {len(orphan_lbls)} label file(s) with no image")
            for s in list(orphan_lbls)[:3]:
                print(f"       {s}")

        # Bounding box sanity check
        bad_boxes = 0
        if lbl_dir.is_dir():
            for lbl_path in lbl_dir.glob("*.txt"):
                try:
                    lines = lbl_path.read_text().strip().splitlines()
                    for line in lines:
                        parts = line.split()
                        if len(parts) < 5:
                            bad_boxes += 1
                            continue
                        cx, cy, w, h = (float(p) for p in parts[1:5])
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and
                                0 < w <= 1 and 0 < h <= 1):
                            bad_boxes += 1
                except Exception:
                    bad_boxes += 1

        if bad_boxes:
            print(f"    [!]  {bad_boxes} bounding box(es) with out-of-range coordinates")
        elif len(lbl_files) > 0:
            print("    Bounding boxes OK OK")

    if not issues:
        print("\n  No annotation issues found. OK")


# =============================================================================
# SECTION 4: Augmentation preview
# =============================================================================

def save_augmentation_preview(data_dir: Path, out_dir: Path, n_samples: int = 4) -> None:
    """Save a grid of original + augmented images."""
    if not HAS_PIL:
        print("  [skip] PIL not installed - skipping augmentation preview")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib not installed - skipping augmentation preview")
        return

    # Find sample images
    samples: list[Path] = []
    train_dir = data_dir / "train"
    if train_dir.is_dir():
        for cls_dir in sorted(train_dir.iterdir()):
            if cls_dir.is_dir():
                imgs = [
                    f for f in cls_dir.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                ]
                if imgs:
                    samples.append(imgs[0])
                if len(samples) >= n_samples:
                    break

    if not samples:
        print("  [skip] No images found for augmentation preview")
        return

    def random_augment(img: "PILImage.Image") -> "PILImage.Image":
        """Apply simple augmentations without TF."""
        import random
        from PIL import ImageEnhance, ImageFilter
        # Random rotation
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, expand=False, fillcolor=0)
        # Random contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.75, 1.25))
        return img

    fig, axes = plt.subplots(len(samples), 2, figsize=(6, 3 * len(samples)))
    if len(samples) == 1:
        axes = [axes]

    for i, img_path in enumerate(samples):
        with PILImage.open(img_path) as im:
            im_gray = im.convert("L").resize((224, 224))
        axes[i][0].imshow(im_gray, cmap="gray")
        axes[i][0].set_title(f"Original\n{img_path.parent.name}")
        axes[i][0].axis("off")

        aug = random_augment(im_gray)
        axes[i][1].imshow(aug, cmap="gray")
        axes[i][1].set_title("Augmented")
        axes[i][1].axis("off")

    fig.suptitle("Augmentation Preview (PIL-based approximation)", fontsize=12)
    out_path = out_dir / "augmentation_preview.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# =============================================================================
# SECTION 5: Model evaluation (requires TensorFlow)
# =============================================================================

def evaluate_model(
    model_path: str,
    class_names_path: str | None,
    data_dir: Path,
    batch_size: int = 32,
) -> None:
    print(SEP)
    print("MODEL EVALUATION")
    print(SEP)

    try:
        import tensorflow as tf
    except ImportError:
        print("  [skip] TensorFlow not installed - skipping model evaluation")
        return

    try:
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError:
        print("  [skip] scikit-learn not installed - skipping model evaluation")
        return

    # Load model
    print(f"  Loading model: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (NotImplementedError, ValueError) as e:
        print(f"  Note: Standard load failed ({type(e).__name__}), using safe_mode=False ...")
        model = tf.keras.models.load_model(model_path, safe_mode=False, compile=False)

    # Load dataset
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_loader import load_datasets  # noqa: E402

    _, _, test_ds, class_names = load_datasets(
        str(data_dir),
        image_size=(224, 224),
        batch_size=batch_size,
        color_mode="grayscale",
        augment=False,
        seed=42,
        verify_class_order=True,
        verbose=False,
    )

    # Override class_names from JSON if provided
    if class_names_path and Path(class_names_path).exists():
        loaded = json.loads(Path(class_names_path).read_text(encoding="utf-8"))
        if isinstance(loaded, list) and loaded != class_names:
            print(f"  Warning: class_names.json={loaded} vs dataset={class_names}")

    num_classes = len(class_names)

    # Collect predictions
    print(f"  Running inference on test set ...")
    y_true, y_pred, y_probs = [], [], []
    for x, y in test_ds:
        p = np.array(model.predict(x, verbose=0))
        y_probs.append(p)
        y_pred.extend(np.argmax(p, axis=1).tolist())
        y_true.extend(y.numpy().tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_probs = np.vstack(y_probs)

    acc = np.mean(y_true == y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # --- Confusion matrix ---
    print(f"\n  Test samples : {len(y_true)}")
    print(f"  Accuracy     : {acc:.4f} ({acc*100:.2f}%)\n")
    print("  Confusion matrix (rows=true, cols=predicted):")
    header = " " * 16 + "".join(f"{c[:8]:>10}" for c in class_names)
    print(f"  {header}")
    print("  " + "-" * (16 + 10 * num_classes))
    for i, row in enumerate(cm):
        print(f"  {class_names[i][:15]:15} " + "".join(f"{v:>10}" for v in row))

    # --- Per-class report ---
    print("\n  Per-class metrics:")
    report_str = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    for line in report_str.splitlines():
        print(f"  {line}")

    # --- Faulty recall (north star metric) ---
    print()
    print("  " + "*" * 50)
    print("  *  FAULTY CLASS RECALL (NORTH STAR METRIC)")
    print("  " + "*" * 50)

    faulty_names = {"faulty", "broken_cap", "broken_ring", "loose_cap"}
    for i, cls in enumerate(class_names):
        if cls in faulty_names or "faulty" in cls or "broken" in cls or "loose" in cls:
            true_pos = cm[i, i]
            total_true = cm[i, :].sum()
            recall = true_pos / max(1, total_true)
            precision = true_pos / max(1, cm[:, i].sum())
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            print(f"  *  [{cls}]  recall={recall:.4f}  precision={precision:.4f}  f1={f1:.4f}")
            if recall < 0.90:
                print(f"  *  [!]  RECALL {recall*100:.1f}% IS BELOW 90% - NOT SAFE FOR PRODUCTION")
            else:
                print(f"  *  Recall {recall*100:.1f}% meets production threshold (>=90%)")

    print("  " + "*" * 50)

    # --- Confidence distribution ---
    print("\n  Confidence (max softmax prob) distribution:")
    conf = y_probs.max(axis=1)
    for lo, hi in [(0, 0.6), (0.6, 0.8), (0.8, 0.9), (0.9, 1.01)]:
        n = int(((conf >= lo) & (conf < hi)).sum())
        bar = "#" * (n // max(1, len(y_true) // 40))
        print(f"    [{lo:.1f}-{hi:.1f}): {n:5d}  {bar}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset and model evaluation for bottle cap inspection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data_dir",
        default="data/processed/cls_3class_crops",
        help="Root of processed classification dataset (default: data/processed/cls_3class_crops)",
    )
    parser.add_argument(
        "--yolo_dir",
        default=None,
        help="(Optional) Path to raw YOLO dataset to check annotation integrity",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="(Optional) Path to .keras model for full evaluation",
    )
    parser.add_argument(
        "--class_names",
        default=None,
        help="(Optional) Path to class_names.json",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--out_dir",
        default="eval_outputs",
        help="Directory to save charts and previews (default: eval_outputs)",
    )
    parser.add_argument(
        "--no_charts",
        action="store_true",
        help="Skip saving charts (faster, no matplotlib needed)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.is_dir():
        print(f"ERROR: --data_dir not found: {data_dir}")
        sys.exit(1)

    print(SEP)
    print("BOTTLE CAP DATASET EVALUATION")
    print(f"  Dataset : {data_dir.resolve()}")
    if args.model:
        print(f"  Model   : {args.model}")
    print(f"  Outputs : {out_dir.resolve()}")
    print(SEP)

    # Section 1: Distribution
    counts = count_images_per_class(data_dir)
    print_distribution(counts)
    if not args.no_charts:
        print("\n  Saving distribution chart...")
        save_distribution_chart(counts, out_dir)

    # Section 2: Integrity
    print()
    report = check_image_integrity(data_dir)
    print_integrity_report(report)

    # Section 3: YOLO integrity (optional)
    if args.yolo_dir:
        yolo_dir = Path(args.yolo_dir)
        if yolo_dir.is_dir():
            check_yolo_annotations(yolo_dir)
        else:
            print(f"\n  [!]  --yolo_dir not found: {yolo_dir}")

    # Section 4: Augmentation preview
    print()
    print(SEP)
    print("AUGMENTATION PREVIEW")
    print(SEP)
    if not args.no_charts:
        save_augmentation_preview(data_dir, out_dir)
    else:
        print("  Skipped (--no_charts)")

    # Section 5: Model evaluation
    if args.model:
        print()
        evaluate_model(
            model_path=args.model,
            class_names_path=args.class_names,
            data_dir=data_dir,
            batch_size=args.batch_size,
        )
    else:
        print()
        print(SEP)
        print("MODEL EVALUATION: skipped (no --model provided)")
        print(f"  To run: python evaluate_dataset.py --data_dir {args.data_dir} \\")
        print(f"            --model models/v6/cap_classifier_best.keras \\")
        print(f"            --class_names models/v6/class_names.json")
        print(SEP)

    print()
    print(SEP)
    print("EVALUATION COMPLETE")
    print(f"  Outputs saved to: {out_dir.resolve()}")
    print(SEP)


if __name__ == "__main__":
    main()
