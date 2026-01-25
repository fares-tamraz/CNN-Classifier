"""Remap YOLO label class IDs.

This fixes datasets where class ordering differs.
Maps old class IDs to new class IDs.
"""

from pathlib import Path

def remap_labels(dataset_root: Path, id_mapping: dict[int, int]) -> int:
    """Remap class IDs in all label files.
    
    Args:
        dataset_root: Root of YOLO dataset (contains train/valid/test)
        id_mapping: Dict mapping old_id -> new_id
        
    Returns:
        Number of files updated
    """
    updated = 0
    
    for split in ["train", "valid", "test"]:
        labels_dir = dataset_root / split / "labels"
        if not labels_dir.exists():
            continue
        
        for label_file in labels_dir.glob("*.txt"):
            lines = label_file.read_text().strip().split('\n')
            if not lines or lines == ['']:
                continue
            
            new_lines = []
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                
                old_id = int(parts[0])
                new_id = id_mapping.get(old_id, old_id)
                parts[0] = str(new_id)
                new_lines.append(' '.join(parts))
            
            label_file.write_text('\n'.join(new_lines) + '\n')
            updated += 1
    
    return updated

if __name__ == "__main__":
    # Remap yolo11-dataset2
    dataset_root = Path("data/raw/yolo11-dataset2")
    
    # Old IDs: 0=brokenRing, 1=brokencap, 2=goodCap, 3=loosecap, 4=noCap
    # New IDs: 0=Broken Cap, 1=Broken Ring, 2=Good Cap, 3=Loose Cap, 4=No Cap
    id_mapping = {
        0: 1,  # brokenRing -> Broken Ring
        1: 0,  # brokencap -> Broken Cap
        2: 2,  # goodCap -> Good Cap (no change)
        3: 3,  # loosecap -> Loose Cap (no change)
        4: 4,  # noCap -> No Cap (no change)
    }
    
    print(f"Remapping yolo11-dataset2 with ID mapping: {id_mapping}")
    updated = remap_labels(dataset_root, id_mapping)
    print(f"Updated {updated} label files")
