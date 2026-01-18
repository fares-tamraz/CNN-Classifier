"""Prepare classification datasets from a Roboflow/YOLO detection dataset.

Roboflow export (YOLO detection format):
  train/images + train/labels
  valid/images + valid/labels   (sometimes called val)
  test/images  + test/labels
  data.yaml (names list)

This tool converts that into folder-based classification datasets:
  <out_dir>/<dataset_name>/{train,val,test}/<class_name>/*.png

Modes:
  - binary:     good vs faulty (GOOD = only 'Good Cap')
  - multiclass: 5-class (Broken Cap / Broken Ring / Good Cap / Loose Cap / No Cap)
  - fault_only: 4-class (Broken Cap / Broken Ring / Loose Cap / No Cap)

Styles:
  - fullframe: one image -> one label folder
  - crops:     crop bounding boxes -> label folder (usually more robust)

Binary + crops important rule:
  *We never mix labels from the same image.*
  If ANY non-good label exists in that image => the image is treated as faulty and
  we only crop the faulty boxes.

Example:
  python tools/prepare_dataset.py --yolo_root data/raw/bottle-cap-yolo \
    --mode binary --style crops --out_dir data/processed --overwrite --dedupe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from PIL import Image

TARGET_SIZE: Tuple[int, int] = (224, 224)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    xc: float
    yc: float
    w: float
    h: float

    def area(self) -> float:
        return self.w * self.h


def read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split_images_dir(yaml_path: Path, rel_path: str) -> Path:
    """data.yaml stores paths like ../train/images relative to the yaml file."""
    return (yaml_path.parent / rel_path).resolve()


def labels_dir_from_images_dir(images_dir: Path) -> Path:
    # YOLO convention: .../images -> .../labels
    return images_dir.parent / "labels"


def label_path_for_image(labels_dir: Path, img_path: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")


def parse_yolo_label_file(label_path: Path) -> List[YoloBox]:
    if not label_path.exists():
        return []
    text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []

    boxes: List[YoloBox] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        boxes.append(YoloBox(class_id=class_id, xc=xc, yc=yc, w=w, h=h))
    return boxes


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def yolo_to_pixel_bbox(box: YoloBox, img_w: int, img_h: int, pad_frac: float) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized bbox to pixel coords (x1,y1,x2,y2) with padding."""
    xc = box.xc * img_w
    yc = box.yc * img_h
    bw = box.w * img_w
    bh = box.h * img_h

    pad_x = bw * pad_frac
    pad_y = bh * pad_frac

    x1 = int(round(xc - bw / 2 - pad_x))
    y1 = int(round(yc - bh / 2 - pad_y))
    x2 = int(round(xc + bw / 2 + pad_x))
    y2 = int(round(yc + bh / 2 + pad_y))

    x1 = clamp(x1, 0, img_w - 1)
    y1 = clamp(y1, 0, img_h - 1)
    x2 = clamp(x2, 1, img_w)
    y2 = clamp(y2, 1, img_h)

    return x1, y1, x2, y2


def safe_rmtree(path: Path) -> None:
    """Best-effort deletion that behaves better on Windows/OneDrive.

    If deletion fails (files locked), we rename the folder so a new run can proceed.
    """

    if not path.exists():
        return

    def _onerror(func, p, exc_info):
        try:
            os.chmod(p, 0o700)
            func(p)
        except Exception:
            pass

    for _ in range(3):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError:
            time.sleep(0.3)
        except OSError:
            time.sleep(0.3)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(path.name + f"__old_{ts}")
    try:
        path.rename(backup)
        print(f"[WARN] Could not delete '{path}'. Renamed it to '{backup}' instead.")
    except Exception as e:
        raise RuntimeError(f"Could not delete or rename '{path}'. Close anything using it, then retry. ({e})")


def mode_class_names(mode: str, names: List[str]) -> List[str]:
    """Return output class folder names in a stable order."""
    if mode == "binary":
        # Keras alphabetical order => faulty=0, good=1
        return ["faulty", "good"]
    if mode == "multiclass":
        return [slugify(n) for n in names]
    if mode == "fault_only":
        return [slugify(n) for n in names if n != "Good Cap"]
    raise ValueError(f"Unknown mode: {mode}")


def class_id_to_output_name(mode: str, names: List[str], class_id: int) -> Optional[str]:
    label = names[class_id]
    if mode == "binary":
        return "good" if label == "Good Cap" else "faulty"
    if mode == "multiclass":
        return slugify(label)
    if mode == "fault_only":
        return None if label == "Good Cap" else slugify(label)
    raise ValueError(f"Unknown mode: {mode}")


def build_dataset_name(mode: str, style: str) -> str:
    if mode == "binary" and style == "fullframe":
        return "cls_binary_fullframe"
    if mode == "binary" and style == "crops":
        return "cls_binary_crops"
    if mode == "multiclass" and style == "fullframe":
        return "cls_5class_fullframe"
    if mode == "multiclass" and style == "crops":
        return "cls_5class_crops"
    if mode == "fault_only" and style == "fullframe":
        return "cls_faulttype_fullframe"
    if mode == "fault_only" and style == "crops":
        return "cls_faulttype_crops"
    return f"cls_{mode}_{style}"


def open_as_grayscale(img_path: Path) -> Optional[Image.Image]:
    try:
        img = Image.open(img_path)
        img = img.convert("L")  # grayscale
        return img
    except Exception:
        return None


def save_resized_png(gray_img: Image.Image, out_path: Path, size: Tuple[int, int] = TARGET_SIZE) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path = out_path.with_suffix(".png")
    gray_img = gray_img.resize(size, resample=Image.Resampling.BILINEAR)
    gray_img.save(out_path)


def dedupe_dataset(dataset_root: Path) -> int:
    """Remove exact duplicate image files across the dataset.

    Keeps priority: train > val > test.
    Returns number of removed files.
    """

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    removed = 0
    seen: Dict[str, Path] = {}

    def file_hash(p: Path) -> str:
        h = hashlib.md5()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    for split in ["train", "val", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            continue
        for p in sorted(split_dir.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            key = file_hash(p)
            if key in seen:
                p.unlink(missing_ok=True)
                removed += 1
            else:
                seen[key] = p

    return removed


def prepare_fullframe(
    *,
    images_dir: Path,
    labels_dir: Path,
    out_split_dir: Path,
    names: List[str],
    mode: str,
    image_label_rule: str,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = label_path_for_image(labels_dir, img_path)
        boxes = parse_yolo_label_file(label_path)

        if mode == "binary":
            if not boxes:
                out_class = "faulty"  # safest default
            else:
                labels = [names[b.class_id] for b in boxes]
                out_class = "good" if all(l == "Good Cap" for l in labels) else "faulty"
        else:
            if not boxes:
                continue
            if image_label_rule == "largest_box":
                chosen = max(boxes, key=lambda b: b.area()).class_id
            elif image_label_rule == "first":
                chosen = boxes[0].class_id
            else:
                raise ValueError(f"Unknown image label rule: {image_label_rule}")
            out_class = class_id_to_output_name(mode, names, chosen)
            if out_class is None:
                continue

        gray = open_as_grayscale(img_path)
        if gray is None:
            continue

        dst = out_split_dir / out_class / (img_path.stem + ".png")
        save_resized_png(gray, dst)
        counts[out_class] = counts.get(out_class, 0) + 1

    return counts


def prepare_crops(
    *,
    images_dir: Path,
    labels_dir: Path,
    out_split_dir: Path,
    names: List[str],
    mode: str,
    crop_strategy: str,
    pad_frac: float,
    min_box_pixels: int,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = label_path_for_image(labels_dir, img_path)
        boxes = parse_yolo_label_file(label_path)
        if not boxes:
            continue

        # Choose which boxes to crop.
        if mode == "binary":
            faulty_boxes = [b for b in boxes if names[b.class_id] != "Good Cap"]
            good_boxes = [b for b in boxes if names[b.class_id] == "Good Cap"]

            if faulty_boxes:
                out_class_fixed = "faulty"
                pool = faulty_boxes
            else:
                out_class_fixed = "good"
                pool = good_boxes if good_boxes else boxes

            if crop_strategy == "all":
                selected = pool
            elif crop_strategy == "largest":
                selected = [max(pool, key=lambda b: b.area())]
            elif crop_strategy == "first":
                selected = [pool[0]]
            else:
                raise ValueError(f"Unknown crop strategy: {crop_strategy}")
        else:
            if crop_strategy == "all":
                selected = boxes
            elif crop_strategy == "largest":
                selected = [max(boxes, key=lambda b: b.area())]
            elif crop_strategy == "first":
                selected = [boxes[0]]
            else:
                raise ValueError(f"Unknown crop strategy: {crop_strategy}")

        gray = open_as_grayscale(img_path)
        if gray is None:
            continue

        img_w, img_h = gray.size

        for i, box in enumerate(selected):
            if mode == "binary":
                out_class = out_class_fixed
            else:
                out_class = class_id_to_output_name(mode, names, box.class_id)
            if out_class is None:
                continue

            x1, y1, x2, y2 = yolo_to_pixel_bbox(box, img_w, img_h, pad_frac)
            if (x2 - x1) < min_box_pixels or (y2 - y1) < min_box_pixels:
                continue

            crop = gray.crop((x1, y1, x2, y2))
            out_name = f"{img_path.stem}__crop{i:02d}__{out_class}.png"
            dst = out_split_dir / out_class / out_name
            save_resized_png(crop, dst)
            counts[out_class] = counts.get(out_class, 0) + 1

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a YOLO detection dataset to classification datasets.")
    parser.add_argument("--yolo_root", type=str, required=True, help="Path to Roboflow YOLO dataset root (contains data.yaml).")
    parser.add_argument("--yaml", type=str, default=None, help="Path to data.yaml (defaults to <yolo_root>/data.yaml).")
    parser.add_argument("--out_dir", type=str, default="data/processed", help="Where to write processed datasets.")
    parser.add_argument("--mode", choices=["binary", "multiclass", "fault_only"], default="binary")
    parser.add_argument("--style", choices=["fullframe", "crops"], default="crops")

    parser.add_argument(
        "--image_label_rule",
        choices=["largest_box", "first"],
        default="largest_box",
        help="How to assign one label per image for fullframe multiclass modes.",
    )
    parser.add_argument(
        "--crop_strategy",
        choices=["all", "largest", "first"],
        default="largest",
        help="Which boxes to crop per image when using style=crops.",
    )
    parser.add_argument("--pad", type=float, default=0.05, help="Pad fraction around each bbox when cropping.")
    parser.add_argument("--min_box_pixels", type=int, default=16, help="Skip crops smaller than this in width/height.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output dataset folder first if it exists.")
    parser.add_argument("--dedupe", action="store_true", help="Remove exact duplicate images across splits (train>val>test).")

    args = parser.parse_args()

    yolo_root = Path(args.yolo_root).resolve()
    yaml_path = Path(args.yaml).resolve() if args.yaml else (yolo_root / "data.yaml")
    out_dir = Path(args.out_dir).resolve()

    if not yaml_path.exists():
        raise FileNotFoundError(f"Could not find data.yaml at: {yaml_path}")

    data = read_yaml(yaml_path)
    names: List[str] = data.get("names")
    if not names:
        raise RuntimeError("data.yaml is missing 'names'")

    # Roboflow sometimes uses 'valid' instead of 'val'. We'll accept either.
    val_key = "val" if data.get("val") else ("valid" if data.get("valid") else "val")

    split_rel = {
        "train": data.get("train"),
        "val": data.get(val_key),
        "test": data.get("test"),
    }

    dataset_name = build_dataset_name(args.mode, args.style)
    dataset_out = out_dir / dataset_name

    if args.overwrite:
        safe_rmtree(dataset_out)

    dataset_out.mkdir(parents=True, exist_ok=True)

    totals: Dict[str, Dict[str, int]] = {}

    for split, rel in split_rel.items():
        if not rel:
            continue

        images_dir = resolve_split_images_dir(yaml_path, rel)

        # If yaml path doesn't resolve, try relative to yolo_root
        if not images_dir.exists():
            rel_path = Path(rel)
            parts = list(rel_path.parts)
            while parts and parts[0] == "..":
                parts.pop(0)
            alt = (yolo_root / Path(*parts)).resolve()
            if alt.exists():
                images_dir = alt

        labels_dir = labels_dir_from_images_dir(images_dir)

        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

        out_split_dir = dataset_out / split

        if args.style == "fullframe":
            counts = prepare_fullframe(
                images_dir=images_dir,
                labels_dir=labels_dir,
                out_split_dir=out_split_dir,
                names=names,
                mode=args.mode,
                image_label_rule=args.image_label_rule,
            )
        else:
            counts = prepare_crops(
                images_dir=images_dir,
                labels_dir=labels_dir,
                out_split_dir=out_split_dir,
                names=names,
                mode=args.mode,
                crop_strategy=args.crop_strategy,
                pad_frac=args.pad,
                min_box_pixels=args.min_box_pixels,
            )

        totals[split] = counts

    if args.dedupe:
        removed = dedupe_dataset(dataset_out)
        if removed:
            print(f"[INFO] Removed {removed} exact-duplicate files across splits.")

    meta = {
        "source_yolo_root": str(yolo_root),
        "source_yaml": str(yaml_path),
        "mode": args.mode,
        "style": args.style,
        "target_size": list(TARGET_SIZE),
        "class_names": mode_class_names(args.mode, names),
        "roboflow_class_names": names,
    }
    (dataset_out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nWrote dataset: {dataset_out}")
    print("Counts per split:")
    for split, c in totals.items():
        pretty = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        print(f"  {split}: {pretty}")


if __name__ == "__main__":
    main()
