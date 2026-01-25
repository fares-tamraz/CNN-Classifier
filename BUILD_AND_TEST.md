# Build a Model & Test (Including Laptop Camera)

End-to-end: prepare data → train → evaluate → test with your **laptop camera**.

**Run all commands from the project root** (the `CNN-Classifier` folder).  
On Windows PowerShell you can use `;` instead of `^` if you split lines.

---

## 0. Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

---

## 1. Put Your YOLO Data in Place

You need a YOLO-style dataset (e.g. from Roboflow or your yolov11 export):

```
data/raw/<your-dataset>/
  data.yaml
  train/images/   train/labels/
  valid/images/   valid/labels/   (or val/)
  test/images/    test/labels/
```

Replace `<your-dataset>` with your folder name (e.g. `bottle-cap-yolo` or `yolov11`).  
`data.yaml` must define `names` and paths for `train` / `val` (or `valid`) / `test`.

---

## 2. Prepare the Classification Dataset

Converts YOLO detection → folder-based classification (binary good/faulty, crops):

```bash
python tools/prepare_dataset.py --yolo_root data/raw/<your-dataset> --mode binary --style crops --out_dir data/processed --overwrite --dedupe
```

Replace `<your-dataset>` with your folder (e.g. `yolov11` or `bottle-cap-yolo`). Example:

```bash
python tools/prepare_dataset.py --yolo_root data/raw/yolov11 --mode binary --style crops --out_dir data/processed --overwrite --dedupe
```

**Output:** `data/processed/cls_binary_crops/` with `train/good`, `train/faulty`, `val/`, `test/`.

---

## 3. Train the Binary Model

```bash
python src/train.py --data_dir data/processed/cls_binary_crops --epochs 20 --augment
```

Defaults: `--model_type mobilenetv2`, `--out_dir models`.  
**Output:** `models/cap_classifier_best.keras`, `models/cap_classifier.keras`, `models/class_names.json`.

For a lighter model (faster, less accurate):

```bash
python src/train.py --data_dir data/processed/cls_binary_crops --model_type simple --epochs 25 --augment
```

---

## 4. (Optional) Evaluate on Test Set

```bash
python src/evaluate.py ^
  --model models/cap_classifier_best.keras ^
  --class_names models/class_names.json ^
  --data_dir data/processed/cls_binary_crops
```

Use `--threshold` to try different p_good cutoffs. For tuning, use `tools/tune_threshold.py` on `--split val`.

---

## 5. Test with Your Laptop Camera

**Laptop camera is fine for testing the pipeline** (capture → preprocess → model → decision → chute).  
**Caveat:** You trained on X‑ray/grayscale bottle frames. A normal RGB webcam is a different domain, so **real-world accuracy on the conveyor will differ**. Use the cam to verify the code runs and that you get accept/reject/chute output.

### 5.1 Realtime sorter (recommended)

Opens the camera, shows a live view. **SPACE** = classify current frame, **Q** = quit.  
By default it’s **dry run**: no hardware, just prints `[BELT]` / `[CHUTE]` so you can test safely.

```bash
python src/realtime_sorter.py --model models/cap_classifier_best.keras --class_names models/class_names.json --show_roi
```

- `--camera 0` is default (usually laptop cam). Use `--camera 1` etc. if you have multiple.
- `--show_roi` draws the crop region. Adjust with `--roi x1 x2 y1 y2` (fractions 0–1). Put the cap in that box before pressing SPACE.
- To actually drive hardware later, use `--no_dry_run` (only when connected to real belt/servos).

### 5.2 Single image (no camera)

If you have a test image (e.g. from your dataset):

```bash
python -m src.cap_classifier --model models/cap_classifier_best.keras --class_names models/class_names.json path/to/image.png
```

Prints JSON: `decision`, `p_good`, `chute_id`, etc.

---

## 6. Quick Reference

| Step | Command |
|------|---------|
| Prepare | `python tools/prepare_dataset.py --yolo_root data/raw/<folder> --mode binary --style crops --out_dir data/processed --overwrite --dedupe` |
| Train | `python src/train.py --data_dir data/processed/cls_binary_crops --epochs 20 --augment` |
| Evaluate | `python src/evaluate.py --model models/cap_classifier_best.keras --class_names models/class_names.json --data_dir data/processed/cls_binary_crops` |
| Laptop cam | `python src/realtime_sorter.py --model models/cap_classifier_best.keras --class_names models/class_names.json --show_roi` |
| Single image | `python -m src.cap_classifier --model ... --class_names ... path/to/image.png` |

---

## 7. Two-Stage (Good vs Faulty → Fault Type)

For multi-chute (Good | Broken Cap | Broken Ring | Loose Cap):

1. **Stage A (binary):** same as above → `models/stageA_binary.keras` + `stageA_classes.json`.
2. **Stage B (fault-only):**  
   Prepare: `--mode fault_only --style crops` → `cls_faulttype_crops/`.  
   Train a second model on that, save as `stageB_faulttype.keras` + `stageB_classes.json`.
3. Run realtime sorter with `--stageB_model` and `--stageB_classes`, or use `CapClassifier` with those paths.

See `CONVEYOR_DESIGN.md` and the README for details.
