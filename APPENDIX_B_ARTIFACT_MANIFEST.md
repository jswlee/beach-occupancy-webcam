# Appendix B: Artifact manifest

_Paper appendix, moved out of `PAPER.md` into the repository so the paper itself stays free of code and variable names. Paths are relative to the repository root unless noted._

| Artifact | Location | Description |
|---|---|---|
| Seg weights | `trained_models/segformer-waterline-v11i/best_model` (HF format) | produced by `train_fast.py`; ONNX export alongside |
| Det weights | YOLO26l-P2 `best.pt`, run `yolo26l_p2_beach_detection_20260524_162410` | md5 c2246c8e…fc90 |
| Seg outputs | `data/mask_stats_clean.csv`, `data/waterline_stats.csv`, `data/night.txt`, `data/people_zones.csv` | per-frame class stats, waterline, night list, zone counts |
| Det outputs | `data/detections.csv`, `data/night_detections.txt` | 284,325 boxes; night list byte-identical to seg |
| Weather / external | `data/external/` | HCDP 5-min obs, NOAA tide, DOH advisories, BWA periods |
| Det run records | `detection/results/` | args.yaml, training_summary.txt, results.csv, merged_metrics.json, annotation_analysis/ |
| Seg methodology | `segmentation/waterline_methodology.md` | operational detail beyond this paper |
| Analysis | `analysis/beach_capacity_analysis.ipynb` | reproduces every figure and statistic in this paper |
