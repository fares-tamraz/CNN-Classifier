"""Remove exact duplicate images from a folder-based classification dataset.

We hash file *bytes* (exact matches). If the same image appears in multiple
splits, that can silently inflate validation/test accuracy (data leakage).

This tool keeps the first occurrence according to a priority order
(train > val > test by default) and deletes later duplicates.

Usage:
  python tools/deduplicate_dataset.py --data_dir data/processed/cls_binary_fullframe
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_images(root: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def md5_bytes(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def split_of(path: Path, data_dir: Path) -> str:
    # expects <data_dir>/<split>/<class>/<file>
    try:
        return path.relative_to(data_dir).parts[0]
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description="Delete exact duplicate images from a classification dataset")
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--priority", type=str, default="train,val,test", help="Keep duplicates in this split order")
    ap.add_argument("--dry_run", action="store_true", help="Only print what would be deleted")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(data_dir)

    priority = [s.strip() for s in args.priority.split(",") if s.strip()]
    pr: Dict[str, int] = {s: i for i, s in enumerate(priority)}

    files = list(iter_images(data_dir))
    print(f"Scanning {len(files)} images under {data_dir}")

    # Map: hash -> (best_path, best_rank)
    best: Dict[str, Tuple[Path, int]] = {}
    deletes: List[Path] = []

    for p in files:
        s = split_of(p, data_dir)
        rank = pr.get(s, 999)
        h = md5_bytes(p)

        if h not in best:
            best[h] = (p, rank)
            continue

        keep_p, keep_rank = best[h]
        if rank < keep_rank:
            # Current one has higher priority => delete old, keep current
            deletes.append(keep_p)
            best[h] = (p, rank)
        else:
            deletes.append(p)

    # De-dupe list (could include repeats if a hash appears >2 times)
    deletes = sorted(set(deletes))

    if not deletes:
        print("No exact duplicates found.")
        return

    print(f"Found {len(deletes)} duplicate files to delete.")

    if args.dry_run:
        for p in deletes[:50]:
            print("[DRY]", p)
        if len(deletes) > 50:
            print(f"... and {len(deletes) - 50} more")
        return

    for p in deletes:
        try:
            p.unlink()
        except Exception as e:
            print(f"Could not delete {p}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
