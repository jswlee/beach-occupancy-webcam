# Appendix A: Reproduction

_Paper appendix, moved out of `PAPER.md` into the repository so the paper itself stays free of code and variable names. All commands run from the repository root (https://github.com/jswlee/ai-beach-carrying-capacity)._

```bash

# Segmentation: labels → train → infer → waterline
python preprocess_masks.py && python train_fast.py
python test_viz_segformer.py --model trained_models/segformer-waterline-v11i/best_model \
    --dataset-dir "D:/Kaanapali" --save-masks --workers 4 --output-dir Kaanapali_full_segformer_v2
python extract_waterline.py

# Detection: preprocess → train (yolo26l-p2) → fused-mask inference
python training/preprocess_and_train.py --enable-slicing --person-only --p2 --model yolo26l.pt
python training/run_tiled_inference.py --model <best.pt> --input-dir "D:/Kaanapali" \
    --seg-mask --save-img --save-img --max-save-img 10 --save-day-only --device cuda --conf 0.20 \
    --output-dir kaanapali_segmask_inference

# Fusion → people_zones.csv; analytics → analysis/beach_capacity_analysis.ipynb
python segmentation/count_people_zones.py --detections <det_out>/detections.csv \
    --masks-dir Kaanapali_full_segformer_v2/masks --output Kaanapali_full_segformer_v2/people_zones.csv
```
