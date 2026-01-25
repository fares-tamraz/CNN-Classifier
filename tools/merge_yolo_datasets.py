"""Merge multiple YOLO detection datasets into one combined dataset.

Each source dataset should have:
  train/images + train/labels
  val/images + val/labels
  test/images + test/labels
  data.yaml

Merges them into a single combined YOLO dataset while preserving all boxes.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import yaml


def merge_yolo_datasets(source_dirs: List[Path], output_dir: Path, overwrite: bool = False) -> None:
    """Merge multiple YOLO datasets into one."""
    
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output dir exists: {output_dir}. Use --overwrite to replace.")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Merge data.yaml from first source
    first_yaml = None
    all_names = None
    
    # Support both 'val' and 'valid' folder names
    split_names = {
        "train": "train",
        "val": ["val", "valid"],  # Try val first, then valid
        "test": "test",
    }
    
    for split_key in ["train", "val", "test"]:
        split_out = output_dir / split_key
        split_out.mkdir(exist_ok=True)
        (split_out / "images").mkdir(exist_ok=True)
        (split_out / "labels").mkdir(exist_ok=True)
        
        image_count = 0
        label_count = 0
        
        for src_idx, src_dir in enumerate(source_dirs):
            # Find the actual split folder (handle both 'val' and 'valid')
            possible_names = split_names[split_key]
            if isinstance(possible_names, str):
                possible_names = [possible_names]
            
            split_src = None
            for name in possible_names:
                candidate = src_dir / name
                if candidate.exists():
                    split_src = candidate
                    break
            
            if not split_src or not split_src.exists():
                print(f"  [skip] {src_dir.name}: no {'/'.join(possible_names)}/ folder")
                continue
            
            images_src = split_src / "images"
            labels_src = split_src / "labels"
            
            if not images_src.exists() or not labels_src.exists():
                continue
            
            # Copy images with unique naming to avoid conflicts
            for img_file in images_src.glob("*"):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                
                # Rename to include dataset index: image.jpg -> image_ds{N}.jpg
                stem = img_file.stem
                new_name = f"{stem}_ds{src_idx}{img_file.suffix}"
                dst = (split_out / "images") / new_name
                shutil.copy2(img_file, dst)
                image_count += 1
            
            # Copy and rename labels to match image names
            for label_file in labels_src.glob("*.txt"):
                stem = label_file.stem
                new_name = f"{stem}_ds{src_idx}.txt"
                dst = (split_out / "labels") / new_name
                shutil.copy2(label_file, dst)
                label_count += 1
        
        print(f"{split_key}: copied {image_count} images, {label_count} labels")
    
    # Create merged data.yaml
    # Use class names from first source
    for src_dir in source_dirs:
        yaml_path = src_dir / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
            if data and "names" in data:
                all_names = data["names"]
                first_yaml = data
                break
    
    if not all_names:
        raise RuntimeError("Could not find class names in any source data.yaml")
    
    merged_data = {
        "path": str(output_dir.resolve()),
        "train": str((output_dir / "train" / "images").resolve()),
        "val": str((output_dir / "val" / "images").resolve()),
        "test": str((output_dir / "test" / "images").resolve()),
        "nc": len(all_names),
        "names": all_names,
    }
    
    yaml_out = output_dir / "data.yaml"
    with open(yaml_out, "w") as f:
        yaml.dump(merged_data, f, default_flow_style=False)
    
    print(f"\nMerged data.yaml written to: {yaml_out}")
    print(f"Classes ({len(all_names)}): {all_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple YOLO datasets")
    parser.add_argument(
        "--sources",
        type=str,
        nargs="+",
        required=True,
        help="Paths to source YOLO dataset roots (each should have train/val/test and data.yaml)",
    )
    parser.add_argument("--output", type=str, required=True, help="Output merged dataset path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    
    args = parser.parse_args()
    
    source_dirs = [Path(s).resolve() for s in args.sources]
    output_dir = Path(args.output).resolve()
    
    print(f"Merging {len(source_dirs)} datasets...")
    for src in source_dirs:
        print(f"  - {src}")
    print(f"Output: {output_dir}\n")
    
    merge_yolo_datasets(source_dirs, output_dir, overwrite=args.overwrite)
    
    print("\n[OK] Merge complete!")


if __name__ == "__main__":
    main()
