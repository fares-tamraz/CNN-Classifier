# Conveyor Belt Inspection – Design & Improvements

Brainstorm and improvement notes for **real-time bottle cap inspection** on a conveyor: camera → pause → scan → chute by output (Good / Broken Cap / Broken Ring / Loose Cap).

---

## 0. Who Owns What (ML vs Mechanical vs Electrical)

| Role | Owns |
|------|------|
| **Mechanical** | Conveyor, chutes, 3D-printed parts, servos/mechanisms, physical layout. |
| **Electrical** | Camera, sensors (trigger), wiring, servo drivers, microcontroller/Arduino. Trigger → pause belt → capture frame → **call ML** → drive diverter from **chute_id**. |
| **You (ML/OpenCV)** | **Good vs faulty** (and optionally fault type). Train models, tune thresholds, and expose a **single inference API**: *image in → decision + chute_id out*. |

**Your deliverable:** Trained model(s) + **`classify_cap`** API. Electrical calls it when they have a frame (e.g. from OpenCV or their camera SDK). You don’t handle hardware, capture, or servos—just the “is it good or faulty?” logic.

See **§ Handoff to electrical** below for the exact I/O contract and how to call the API.

---

## 1. What You’ve Got Right

- **Two-stage design**: Good vs faulty first, then fault-type (broken_cap, broken_ring, loose_cap) only when you reject. Matches real factory flow and keeps Stage B focused.
- **Binary + fault-only datasets**: `prepare_dataset` supports `binary` (good/faulty) and `fault_only` (4-class) for Stage A and Stage B. Solid.
- **Crops over full-frame**: Training on YOLO crops (cap region) is usually more robust than full conveyor images.
- **Accept / Reject / Manual-review band**: Threshold band (`th_accept`, `th_reject`) instead of a single cutoff is the right idea for inspection.
- **`realtime_sorter`**: Already uses OpenCV, ROI crop, grayscale preprocessing, and a `BeltController` placeholder. Good base.

---

## 2. OpenCV vs “Something Better”

**Use OpenCV (cv2) for capture.** It’s the right default for:

- USB webcams
- Many USB industrial cameras
- Generic capture when you don’t have a vendor SDK

**When to use something else:**

- **GigE / CoaXPress / GenICam**: Use the camera’s SDK (e.g. Basler pypylon, FLIR Spinnaker) for trigger control, exposure, etc. You can still hand `numpy` frames to your preprocessing + model.
- **X-ray imaging**: If the actual line uses an X-ray sensor, you’ll use that vendor’s API to grab frames. Your CNN stays the same; only the capture source changes.

So: **OpenCV for “film the cap”** is fine. Add `opencv-python` to `requirements.txt` (done). If you later move to a specific industrial or X-ray camera, swap the capture layer only.

---

## 3. Conveyor Flow: Trigger → Pause → Scan → Chute

You want: **belt runs → cap in view → pause → scan → chute by result**.

### 3.1 Triggering (when to capture)

| Method | Pros | Cons |
|--------|------|------|
| **Photoelectric / proximity sensor** | Reliable, precise “cap in zone” | Extra hardware, wiring |
| **Encoder on belt** | Known position, repeatable | Need encoder + integration |
| **Software (motion / detection)** | No extra hardware | Less repeatable, timing jitter |

**Recommendation:** Use a **sensor** (or encoder) to trigger “cap in scan zone” → **pause belt** → **grab frame** → run model → **divert to chute** → resume. OpenCV does the “grab frame” part; the trigger drives when you capture.

### 3.2 Scan zone

- **Fixed ROI**: Camera mounted above (or at angle) with fixed FOV. Define ROI (e.g. `--roi` in `realtime_sorter`) so you always crop the same “cap region”. Reduces clutter and improves consistency.
- **Dedicated scan zone**: Cap enters zone → belt stops (or creeps) → capture. Avoids motion blur and keeps cap position stable.

### 3.3 Chutes

- **2 chutes**: Accept (good) | Reject (all faults). Simplest.
- **4 chutes**: Good | Broken Cap | Broken Ring | Loose Cap. Better for analytics and different rework.

Your two-stage setup supports both: Stage A → good → chute 1; Stage A faulty → Stage B → chutes 2–4 by fault type.

---

## 4. X‑ray Training vs RGB Camera (Domain Gap)

**Important:** Your training data is **X‑ray / grayscale** “bottle frames”.  

- If the **live line** uses the **same** X‑ray (or similar) imaging: domain gap is small. Your current grayscale pipeline is appropriate.
- If the **live line** uses a **normal RGB camera** (visible light): domain gap is large. The model may not generalize well.

**Options if you go RGB on the line:**

1. **Collect new training data** from the actual RGB camera (and optionally same conveyor/lighting), then re-train.
2. **Domain adaptation** (e.g. train on X‑ray, adapt to RGB) — more involved.
3. **Stick with X‑ray (or similar) on the line** so training and deployment match.

Worth deciding up front: **same modality** (X‑ray-ish) vs **RGB** on the conveyor.

---

## 5. Lighting & Environment

- **Consistent lighting**: Avoid shadows, reflections, and time-of-day changes. Helps both classical preprocessing and the CNN.
- **Same as (or close to) training**: If you can, mimic training conditions (layout, distance, lighting) when you collect validation data from the line.

---

## 6. What Might Feel “Ruined” or Overcomplicated

- **Many prep modes**: `binary` / `multiclass` / `fault_only` × `fullframe` / `crops` can feel like a maze. For your flow, you only need:
  - **Stage A**: `binary` + `crops` (good vs faulty).
  - **Stage B**: `fault_only` + `crops` (broken_cap, broken_ring, loose_cap, no_cap).
- **Small bugs**: `evaluate` used `verify_classes` (wrong arg) and `train` used `simple_cnn` while `model` expects `simple`. Those are fixed.
- **Thresholds**: Use `tools/tune_threshold.py` on validation to pick `th_accept` / `th_reject` (and Stage A threshold) from metrics, not by hand.

---

## 7. Concrete Improvements

### 7.1 Already done

- Fixed `evaluate.py`: `verify_class_order` and correct `load_datasets` usage.
- Fixed `train.py`: `--model_type simple` (not `simple_cnn`).
- Added `opencv-python` to `requirements.txt`.

### 7.2 Realtime sorter

- **Two-stage mode**: Optionally load Stage A + Stage B. Flow: Stage A → good → Accept; faulty → Stage B → fault type.
- **Multi-chute**: Extend `BeltController` (or equivalent) with e.g. `divert(chute_id)`: 0 = good, 1–3 = fault types. Keep dry-run mode (print only) for safety.
- **Hardware trigger**: Optional GPIO or serial input “cap in zone” to trigger capture instead of SPACE. Enables full conveyor integration.

### 7.3 Operational

- **Calibration**: Store ROI (and optionally camera index) per setup. Reuse in `realtime_sorter` via config.
- **Tune thresholds**: Run `tune_threshold` on val (and test) for Stage A; set `th_accept` / `th_reject` from that.
- **Logging**: Keep logging decisions (and optionally images) for rejects/unsure for later analysis and model updates.

---

## 8. Summary

| Topic | Recommendation |
|-------|----------------|
| **Capture** | OpenCV; switch to camera/X‑ray SDK only if you use such hardware. |
| **Trigger** | Prefer sensor (or encoder) → pause → capture → infer → chute. |
| **Pipeline** | Two-stage (good vs faulty → fault type) + multi-chute. |
| **Training** | Binary + fault-only crops; match imaging modality (X‑ray vs RGB) to the line. |
| **Thresholds** | Use `tune_threshold`; set accept/reject from val/test metrics. |

Next step: extend `realtime_sorter` for **two-stage + multi-chute** and (optionally) **triggered capture**, then connect it to your conveyor hardware (servos, sensors) via `BeltController` or a small hardware abstraction layer.

---

## 9. Handoff to Electrical (ML/OpenCV API)

You provide a **Python API** that electrical can call when they have a frame. Everything else (camera, trigger, servos) is their side.

### 9.1 `CapClassifier` and `classify_cap`

- **`src/cap_classifier.py`**: Holds loaded model(s), preprocessing, and thresholds.
- **`CapClassifier(model_path, class_names_path, ...)`**: Load once. Then **`clf.classify(image)`** per frame.
- **`classify_cap(image, model_path, class_names_path, ...)`**: One-shot helper (loads models each call; use `CapClassifier` in a loop).

### 9.2 Input

- **`image`**: `str` or `Path` (file path) **or** `np.ndarray` (BGR, `H×W×3`), e.g. from `cv2.imread` / `cv2.VideoCapture.read()`.
- **`roi`**: `(x1, x2, y1, y2)` as fractions of width/height. Default `(0, 1, 0, 1)` = full frame. Match camera layout (e.g. cap in center).

### 9.3 Output (dict)

| Key | Type | Meaning |
|-----|------|--------|
| `decision` | `"good"` \| `"faulty"` \| `"unsure"` | From threshold band. |
| `p_good` | `float` | Model confidence for “good”. |
| `chute_id` | `int` | **0** = good, **1** = reject (binary) or fault-type chute 1, **2–4** = fault types when Stage B used. |
| `fault_type` | `str` \| `None` | e.g. `"loose_cap"` when Stage B runs; `None` otherwise. |
| `fault_confidence` | `float` \| `None` | Stage B confidence when applicable. |

Electrical uses **`chute_id`** to drive the diverter (e.g. servo to chute 0, 1, 2, …).

### 9.4 Example (electrical side)

```python
from pathlib import Path
import cv2
from src.cap_classifier import CapClassifier

# Load once at startup
clf = CapClassifier(
    "models/cap_classifier_best.keras",
    "models/class_names.json",
    roi=(0.3, 0.7, 0.15, 0.55),  # match your camera ROI
    th_accept=0.6,
    th_reject=0.4,
)

# When sensor says "cap in zone": capture frame, then:
cap = cv2.VideoCapture(0)
ok, frame = cap.read()
result = clf.classify(frame)

print(result["decision"], result["chute_id"])  # e.g. "good" 0  or  "faulty" 2
# → Drive servo to chute result["chute_id"], then advance belt.
```

### 9.5 What you don’t do

- Camera choice, trigger hardware, or capture loop.
- Servo or belt control.
- Mechanical layout of conveyor/chutes.

You only guarantee: **same image format (BGR, ROI) as agreed → correct `decision` and `chute_id`** from your trained models.
