#!/usr/bin/env python3
"""Collect and print all key metrics for the latest model checkpoint.

Run this to populate the .milestones/model_snapshot.md with real numbers.

Usage:
  python metrics_collector.py
  python metrics_collector.py --model models/v6/cap_classifier_best.keras
  python metrics_collector.py --model models/v4/cap_classifier_best.keras \\
    --data_dir data/processed/cls_5class_crops_v3

Output is printed to stdout AND saved to .milestones/metrics_latest.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect model metrics")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to .keras model (default: auto-detect latest)",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Path to classification dataset root (default: auto-detect from model version)",
    )
    parser.add_argument(
        "--class_names",
        default=None,
        help="Path to class_names.json (default: auto-detect near model)",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Install with: pip install tensorflow scikit-learn")
        sys.exit(1)

    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from data_loader import load_datasets  # noqa: E402

    # Auto-detect model
    model_path = args.model
    if model_path is None:
        for candidate in [
            "models/v6/cap_classifier_best.keras",
            "models/v5/cap_classifier_best.keras",
            "models/v4/cap_classifier_best.keras",
        ]:
            if Path(candidate).exists():
                model_path = candidate
                break
    if model_path is None:
        print("ERROR: No model found. Use --model to specify path.")
        sys.exit(1)

    model_dir = Path(model_path).parent
    version = model_dir.name  # e.g. "v6"

    # Auto-detect class_names.json
    class_names_path = args.class_names
    if class_names_path is None:
        for candidate in [
            model_dir / "class_names.json",
            Path("models/class_names.json"),
        ]:
            if candidate.exists():
                class_names_path = str(candidate)
                break

    # Auto-detect data_dir
    data_dir = args.data_dir
    if data_dir is None:
        if version in {"v5", "v6"}:
            data_dir = "data/processed/cls_3class_crops"
        else:
            data_dir = "data/processed/cls_5class_crops_v3"

    if not Path(data_dir).is_dir():
        print(f"ERROR: Dataset not found: {data_dir}")
        sys.exit(1)

    print("=" * 70)
    print("METRICS COLLECTOR")
    print("=" * 70)
    print(f"  Model      : {model_path}")
    print(f"  Dataset    : {data_dir}")
    print(f"  Class names: {class_names_path}")

    # Load model
    print("\nLoading model...")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (NotImplementedError, ValueError):
        model = tf.keras.models.load_model(model_path, safe_mode=False, compile=False)

    # Model info
    total_params = model.count_params()
    trainable_params = sum(
        np.prod(w.shape) for w in model.trainable_weights
    )
    input_shape = model.input_shape  # e.g. (None, 224, 224, 1)
    output_shape = model.output_shape  # e.g. (None, 3)
    num_layers = len(model.layers)

    # Load dataset
    _, _, test_ds, class_names = load_datasets(
        data_dir,
        image_size=(224, 224),
        batch_size=args.batch_size,
        color_mode="grayscale",
        augment=False,
        seed=42,
        verify_class_order=True,
        verbose=False,
    )
    num_classes = len(class_names)

    # Inference with timing
    print("Running inference on test set...")
    y_true, y_pred, y_probs = [], [], []
    t0 = time.perf_counter()
    n_images = 0

    for x, y in test_ds:
        p = np.array(model.predict(x, verbose=0))
        y_probs.append(p)
        y_pred.extend(np.argmax(p, axis=1).tolist())
        y_true.extend(y.numpy().tolist())
        n_images += len(y)

    elapsed = time.perf_counter() - t0

    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    y_probs = np.vstack(y_probs)

    ms_per_image = (elapsed / max(1, n_images)) * 1000
    fps = n_images / max(1e-6, elapsed)

    # Metrics
    acc = float(np.mean(y_true == y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4, output_dict=True
    )

    # Per-class P/R/F1
    faulty_variants = {"faulty", "broken_cap", "broken_ring", "loose_cap"}
    faulty_metrics: dict[str, dict] = {}
    for cls in class_names:
        if cls in faulty_variants or "faulty" in cls or "broken" in cls or "loose" in cls:
            faulty_metrics[cls] = report.get(cls, {})

    # Build output text
    lines = []
    lines.append("=" * 70)
    lines.append("MODEL METRICS REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("MODEL INFO")
    lines.append(f"  Path            : {model_path}")
    lines.append(f"  Version         : {version}")
    lines.append(f"  Input shape     : {input_shape}")
    lines.append(f"  Output shape    : {output_shape}")
    lines.append(f"  Total layers    : {num_layers}")
    lines.append(f"  Total params    : {total_params:,}")
    lines.append(f"  Trainable params: {trainable_params:,}")
    lines.append("")
    lines.append("DATASET")
    lines.append(f"  Root            : {data_dir}")
    lines.append(f"  Classes         : {class_names}")
    lines.append(f"  Test samples    : {n_images}")
    lines.append("")
    lines.append("PERFORMANCE")
    lines.append(f"  Test accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    lines.append(f"  Inference speed : {ms_per_image:.2f} ms/image  ({fps:.1f} FPS)")
    lines.append("")
    lines.append("PER-CLASS METRICS")
    lines.append(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    lines.append("  " + "-" * 54)
    for cls in class_names:
        m = report.get(cls, {})
        lines.append(
            f"  {cls:<20} {m.get('precision', 0):>10.4f} "
            f"{m.get('recall', 0):>10.4f} "
            f"{m.get('f1-score', 0):>10.4f} "
            f"{int(m.get('support', 0)):>10}"
        )
    lines.append("")
    lines.append("CONFUSION MATRIX (rows=true, cols=predicted)")
    header = " " * 22 + "".join(f"{c[:8]:>10}" for c in class_names)
    lines.append(f"  {header}")
    lines.append("  " + "-" * (22 + 10 * num_classes))
    for i, row in enumerate(cm):
        lines.append(f"  {class_names[i][:20]:20}  " + "".join(f"{v:>10}" for v in row))
    lines.append("")
    lines.append("*" * 50)
    lines.append("*  FAULTY RECALL (NORTH STAR METRIC)")
    lines.append("*" * 50)
    for cls, m in faulty_metrics.items():
        recall = m.get("recall", 0)
        precision = m.get("precision", 0)
        f1 = m.get("f1-score", 0)
        support = int(m.get("support", 0))
        flag = "!! BELOW 90% -- REVIEW BEFORE PRODUCTION" if recall < 0.90 else "OK - meets >=90% threshold"
        lines.append(f"  [{cls}]")
        lines.append(f"    Recall    : {recall:.4f} ({recall*100:.1f}%)  {flag}")
        lines.append(f"    Precision : {precision:.4f}")
        lines.append(f"    F1-score  : {f1:.4f}")
        lines.append(f"    Support   : {support} test samples")
    lines.append("*" * 50)

    output = "\n".join(lines)
    print("\n" + output)

    # Save to .milestones/
    milestone_dir = Path(".milestones")
    milestone_dir.mkdir(exist_ok=True)
    out_file = milestone_dir / "metrics_latest.txt"
    out_file.write_text(output, encoding="utf-8")
    print(f"\nSaved to: {out_file}")


if __name__ == "__main__":
    main()
