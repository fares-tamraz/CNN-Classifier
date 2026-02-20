#!/usr/bin/env python3
"""Create 3-class dataset from 5-class dataset.

Mapping:
- good_cap -> good
- broken_cap, broken_ring, loose_cap -> faulty
- no_cap -> no_cap

This creates symbolic links (Windows junction points) to avoid data duplication.
"""

import argparse
import json
import shutil
from pathlib import Path
from collections import defaultdict


# Class mapping: old_class -> new_class
CLASS_MAPPING = {
    "good_cap": "good",
    "broken_cap": "faulty",
    "broken_ring": "faulty",
    "loose_cap": "faulty",
    "no_cap": "no_cap",
}


def create_3class_dataset(src_dir: str, dst_dir: str, copy: bool = False):
    """
    Create 3-class dataset from 5-class dataset.
    
    Args:
        src_dir: Source 5-class dataset directory
        dst_dir: Destination 3-class dataset directory
        copy: If True, copy files. If False, create hardlinks (faster, saves space).
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    
    if dst_dir.exists():
        print(f"Removing existing directory: {dst_dir}")
        shutil.rmtree(dst_dir)
    
    stats = defaultdict(lambda: defaultdict(int))
    
    for split in ["train", "val", "test"]:
        split_src = src_dir / split
        if not split_src.exists():
            print(f"Warning: {split_src} does not exist, skipping")
            continue
        
        for old_class_dir in sorted(split_src.iterdir()):
            if not old_class_dir.is_dir():
                continue
            
            old_class = old_class_dir.name
            if old_class not in CLASS_MAPPING:
                print(f"Warning: Unknown class {old_class}, skipping")
                continue
            
            new_class = CLASS_MAPPING[old_class]
            dst_class_dir = dst_dir / split / new_class
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in old_class_dir.glob("*.png"):
                dst_path = dst_class_dir / img_path.name
                
                if copy:
                    shutil.copy2(img_path, dst_path)
                else:
                    # Use hardlink to save space
                    try:
                        dst_path.hardlink_to(img_path)
                    except OSError:
                        # Fallback to copy if hardlink fails (different drives)
                        shutil.copy2(img_path, dst_path)
                
                stats[split][new_class] += 1
    
    # Print summary
    print("\n" + "="*60)
    print("3-CLASS DATASET CREATED")
    print("="*60)
    print(f"\nSource: {src_dir}")
    print(f"Destination: {dst_dir}")
    print(f"\nClass mapping:")
    for old, new in CLASS_MAPPING.items():
        print(f"  {old:15s} -> {new}")
    
    print("\n" + "-"*60)
    print("Dataset statistics:")
    print("-"*60)
    
    class_names = sorted(set(CLASS_MAPPING.values()))
    
    for split in ["train", "val", "test"]:
        if split not in stats:
            continue
        total = sum(stats[split].values())
        print(f"\n{split.upper()}:")
        for class_name in class_names:
            count = stats[split][class_name]
            pct = 100 * count / total if total > 0 else 0
            print(f"  {class_name:10s}: {count:5d} ({pct:5.1f}%)")
        print(f"  {'TOTAL':10s}: {total:5d}")
    
    # Grand totals
    grand_total = sum(sum(s.values()) for s in stats.values())
    print(f"\nGrand total: {grand_total} images")
    
    # Save class names
    class_names_path = dst_dir / "class_names.json"
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    print(f"\nClass names saved to: {class_names_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Create 3-class dataset from 5-class")
    parser.add_argument(
        "--src", 
        default="data/processed/cls_5class_crops_v3",
        help="Source 5-class dataset directory"
    )
    parser.add_argument(
        "--dst", 
        default="data/processed/cls_3class_crops",
        help="Destination 3-class dataset directory"
    )
    parser.add_argument(
        "--copy", 
        action="store_true",
        help="Copy files instead of creating hardlinks"
    )
    args = parser.parse_args()
    
    create_3class_dataset(args.src, args.dst, args.copy)


if __name__ == "__main__":
    main()
