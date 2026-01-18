# Bottle Cap X-ray Classifier (CNN)

This repo trains a **computer-vision classifier** to inspect bottle caps using **X-ray-like grayscale images**.

It is designed like a small production pipeline:
1) **Prepare data** (convert Roboflow YOLO detection export into clean classification folders)
2) **Train** (grayscale CNN)
3) **Evaluate** (confusion matrix + precision/recall/F1)
4) **Predict** (confidence-gated decisions + CSV logging)

## Folder layout

```
assets/            # images used in documentation only
src/               # training/eval/inference code
tools/             # one-time dataset preparation scripts
data/raw/          # (not committed) Roboflow download goes here
data/processed/    # (not committed) prepared classification datasets
models/            # (not committed) saved models + class_names.json
logs/              # (not committed) prediction logs
```

## 0) Install

Create a virtual environment, then:

```
pip install -r requirements.txt
```

## 1) Put your Roboflow dataset in `data/raw/`

Example (your downloaded zip extracted):

```
data/raw/bottle-cap-yolo/
  data.yaml
  train/images ... train/labels
  valid/images ... valid/labels
  test/images  ... test/labels
```

## 2) Prepare a classification dataset (automatic)

### Recommended: **binary + crops** (robust on real conveyor footage)

```
python tools/prepare_dataset.py \
  --yolo_root data/raw/bottle-cap-yolo \
  --mode binary \
  --style crops \
  --out_dir data/processed \
  --overwrite
```

Output:

```
data/processed/cls_binary_crops/
  train/{good,faulty}
  val/{good,faulty}
  test/{good,faulty}
  meta.json
```

### Optional: 5-class classification

```
python tools/prepare_dataset.py --yolo_root data/raw/bottle-cap-yolo --mode multiclass --style crops --out_dir data/processed --overwrite
```

Classes are slugified for clean folder names:
`broken_cap, broken_ring, good_cap, loose_cap, no_cap`

### Optional: fault-type only (stage B)

```
python tools/prepare_dataset.py --yolo_root data/raw/bottle-cap-yolo --mode fault_only --style crops --out_dir data/processed --overwrite
```

## 3) Train

Binary example:

```
python src/train.py \
  --data_dir data/processed/cls_binary_crops \
  --epochs 20 \
  --augment \
  --use_class_weights
```

This writes:
- `models/cap_classifier.keras`
- `models/class_names.json`

## 4) Evaluate

```
python src/evaluate.py \
  --data_dir data/processed/cls_binary_crops \
  --model_path models/cap_classifier.keras \
  --class_names models/class_names.json
```

## 5) Predict (confidence gating + logging)

Single image:

```
python src/predict.py \
  --model_path models/cap_classifier.keras \
  --class_names models/class_names.json \
  --image path/to/image.jpg \
  --accept_threshold 0.95
```

Batch folder:

```
python src/predict.py --model_path models/cap_classifier.keras --class_names models/class_names.json --folder path/to/images
```

Predictions are appended to `logs/predictions.csv`.

## Optional: two-stage inspection

Train a binary model (Stage A) and a fault-type model (Stage B), then run:

```
python src/predict_two_stage.py \
  --stageA_model models/stageA_binary.keras \
  --stageA_classes models/stageA_classes.json \
  --stageB_model models/stageB_faulttype.keras \
  --stageB_classes models/stageB_classes.json \
  --image path/to/image.jpg
```


## Notes on "GOOD vs FAULTY"

- GOOD = only **Good Cap**
- FAULTY = Broken Cap / Broken Ring / Loose Cap / No Cap (and missing/empty labels are treated as FAULTY for safety)

## License / Attribution

If your dataset is from Roboflow, cite the dataset author and Roboflow in your report.

