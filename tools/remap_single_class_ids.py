#!/usr/bin/env python3
"""
Remap class IDs in single-class YOLO datasets to match multi-class ID scheme.

Usage:
  python tools/remap_single_class_ids.py --source_dir data/raw/brknring2 --old_class_id 0 --new_class_id 1
  python tools/remap_single_class_ids.py --source_dir data/raw/brknring3 --old_class_id 0 --new_class_id 1
"""

import argparse
import os
from pathlib import Path


def remap_class_ids(source_dir, old_class_id, new_class_id):
    """
    Remap class IDs in all label files under source_dir.
    
    Args:
        source_dir: Path to dataset (contains train/, valid/, test/)
        old_class_id: Original class ID (typically 0 for single-class datasets)
        new_class_id: Target class ID (from multi-class scheme)
    """
    source_path = Path(source_dir)
    
    # Find all label directories
    label_dirs = []
    for split_dir in ['train', 'valid', 'val', 'test']:
        labels_dir = source_path / split_dir / 'labels'
        if labels_dir.exists():
            label_dirs.append(labels_dir)
    
    if not label_dirs:
        print(f"❌ No label directories found in {source_dir}")
        return 0
    
    total_files = 0
    total_lines = 0
    
    for labels_dir in label_dirs:
        print(f"\n📁 Processing {labels_dir.relative_to(source_path.parent)}")
        
        label_files = list(labels_dir.glob('*.txt'))
        if not label_files:
            print(f"   ⚠️  No .txt files found")
            continue
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    continue
                
                # Remap first column (class ID)
                remapped_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    class_id = int(parts[0])
                    if class_id == old_class_id:
                        parts[0] = str(new_class_id)
                    
                    remapped_lines.append(' '.join(parts) + '\n')
                
                # Write back
                with open(label_file, 'w') as f:
                    f.writelines(remapped_lines)
                
                total_files += 1
                total_lines += len(remapped_lines)
                
            except Exception as e:
                print(f"   ❌ Error processing {label_file.name}: {e}")
    
    print(f"\n✅ Remapped {total_files} label files ({total_lines} total lines)")
    print(f"   Class {old_class_id} → Class {new_class_id}")


def main():
    parser = argparse.ArgumentParser(
        description='Remap class IDs in single-class YOLO datasets'
    )
    parser.add_argument(
        '--source_dir',
        required=True,
        help='Path to source dataset (e.g., data/raw/brknring2)'
    )
    parser.add_argument(
        '--old_class_id',
        type=int,
        default=0,
        help='Original class ID to remap from (default: 0)'
    )
    parser.add_argument(
        '--new_class_id',
        type=int,
        required=True,
        help='Target class ID to remap to (e.g., 1 for Broken Ring)'
    )
    
    args = parser.parse_args()
    
    print(f"🔄 Remapping class IDs in {args.source_dir}")
    print(f"   {args.old_class_id} → {args.new_class_id}")
    
    remap_class_ids(
        args.source_dir,
        args.old_class_id,
        args.new_class_id
    )


if __name__ == '__main__':
    main()
