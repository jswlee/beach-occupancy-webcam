"""Benchmark pretrained (non-finetuned) YOLO26n/s/m/l/x on the Beach Counter
v16 test set using the SAME merged tiled evaluation as the production model:
3-bottom-tile inference, cross-tile NMS merge (IoU 0.5), greedy IoU-0.5
matching to person GT, conf >= 0.20.

For COCO base models we keep only class-0 (person) predictions so the
comparison is 'person detector vs person detector' -- other COCO classes
(umbrella, surfboard, ...) would otherwise be counted as false positives
through no fault of the person-detection capability.

Usage:  python eval_baselines.py
Output: detection/results/baseline_metrics.csv + per-model merged_metrics.json
"""

import sys
import json
import csv
from pathlib import Path

# --- cv-training imports (exact production eval code) ---
CV_TRAINING = Path(r"C:\Users\JR\Documents\GitHub\cv-training\training")
sys.path.insert(0, str(CV_TRAINING))
from train_yolo_simple import run_merged_test_inference  # noqa: E402

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

# Same raw test set the production merged eval used (1,695 person GT)
RAW_DATA = Path(r"C:\Users\JR\Documents\GitHub\beach-obj-detection-training"
                r"\roboflow_data\Beach Counter.v16i.yolov8")
OUT_DIR = Path(__file__).parent / "results" / "baselines_v16prod"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Person-only predictions for COCO base models
_orig_predict = YOLO.predict
def _person_only_predict(self, *a, **kw):
    kw.setdefault("classes", [0])
    return _orig_predict(self, *a, **kw)
YOLO.predict = _person_only_predict

MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"]

rows = []
for name in MODELS:
    run_dir = OUT_DIR / name.replace(".pt", "")
    run_dir.mkdir(exist_ok=True)
    print(f"\n########## {name} ##########")
    try:
        run_merged_test_inference(
            model_path=name,               # Ultralytics auto-downloads
            raw_data_dir=str(RAW_DATA),
            img_size=640,                  # same as production eval
            tile_size=640,
            run_dir=run_dir,
            conf=0.20,
            iou_merge=0.50,
            iou_match=0.50,
            person_only=True,
            person_class_id=3,             # person id in raw labels
        )
        s = json.load(open(run_dir / "merged_test_inference"
                           / "merged_metrics.json"))["summary"]
        s["model"] = name
        rows.append(s)
    except Exception as e:
        print(f"FAILED {name}: {e}")
        rows.append({"model": name, "precision": None, "recall": None,
                     "f1": None, "ap50": None})

cols = ["model", "images", "total_gt", "total_tp", "total_fp", "total_fn",
        "precision", "recall", "f1", "ap50"]
with open(OUT_DIR / "baseline_metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print("\n================ BASELINE COMPARISON ================")
for r in rows:
    print(f"{r['model']:12s}  P={r['precision']}  R={r['recall']}  "
          f"F1={r['f1']}  AP50={r['ap50']}")
print("(finetuned yolo26l-P2:  P=0.7095  R=0.6555  F1=0.6814  AP50=0.5958)")
