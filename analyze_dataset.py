#!/usr/bin/env python3
"""Analyze dataset split and distribution."""

import os
from pathlib import Path
from collections import defaultdict
import re


def extract_dataset_tag(filename):
    """Extract ds0, ds1, ds2, ds3 from filename."""
    match = re.search(r'_ds(\d+)_', filename)
    return f"ds{match.group(1)}" if match else "unknown"


def analyze_split(split_dir):
    """Analyze a single split (train/val/test)."""
    stats = defaultdict(lambda: defaultdict(int))
    
    for class_dir in Path(split_dir).iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        
        for img_file in class_dir.glob("*.png"):
            ds_tag = extract_dataset_tag(img_file.name)
            stats[class_name][ds_tag] += 1
    
    return stats


def main():
    data_dir = Path("data/processed/cls_5class_crops_v3")
    
    print("="*80)
    print("DATASET SPLIT ANALYSIS")
    print("="*80 + "\n")
    
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        
        stats = analyze_split(split_dir)
        
        print(f"\n{split.upper()} SPLIT:")
        print("-" * 60)
        
        # Total per class
        for class_name in sorted(stats.keys()):
            total = sum(stats[class_name].values())
            print(f"\n{class_name:15s}: {total:4d} total")
            
            # Breakdown by dataset
            for ds_tag in sorted(stats[class_name].keys()):
                count = stats[class_name][ds_tag]
                pct = 100 * count / total if total > 0 else 0
                print(f"  {ds_tag:10s}: {count:4d} ({pct:5.1f}%)")
    
    # Check for dataset imbalance across splits
    print("\n" + "="*80)
    print("DATASET SOURCE DISTRIBUTION ACROSS SPLITS")
    print("="*80 + "\n")
    
    all_stats = {}
    for split in ["train", "val", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            all_stats[split] = analyze_split(split_dir)
    
    # For each class, show how each dataset is distributed
    all_classes = set()
    for split_stats in all_stats.values():
        all_classes.update(split_stats.keys())
    
    for class_name in sorted(all_classes):
        print(f"\n{class_name.upper()}:")
        print("-" * 60)
        
        # Get all dataset tags for this class
        all_ds_tags = set()
        for split_stats in all_stats.values():
            if class_name in split_stats:
                all_ds_tags.update(split_stats[class_name].keys())
        
        for ds_tag in sorted(all_ds_tags):
            train_count = all_stats.get("train", {}).get(class_name, {}).get(ds_tag, 0)
            val_count = all_stats.get("val", {}).get(class_name, {}).get(ds_tag, 0)
            test_count = all_stats.get("test", {}).get(class_name, {}).get(ds_tag, 0)
            total = train_count + val_count + test_count
            
            if total == 0:
                continue
            
            train_pct = 100 * train_count / total
            val_pct = 100 * val_count / total
            test_pct = 100 * test_count / total
            
            print(f"  {ds_tag:10s}: train={train_count:4d} ({train_pct:5.1f}%), "
                  f"val={val_count:3d} ({val_pct:5.1f}%), "
                  f"test={test_count:3d} ({test_pct:5.1f}%)")


if __name__ == "__main__":
    main()
