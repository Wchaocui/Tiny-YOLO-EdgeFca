# Tiny-YOLO-EdgeFca

Code, pre-trained weights, and the **SLLI** dataset for
**"Tiny-YOLO-EdgeFca: A Lightweight Framework for Low-Light Space Object Detection"**.

> This repository is currently **private** during peer review. It will be made
> public upon acceptance.

## Repository Structure

```
code/
  LeYOLO-main/        Custom Ultralytics fork containing the proposed modules
                      (EdgeFca, MMA-SPPF/TOPKSPP, lightweight backbone) and
                      the SLLI screening scripts (dataset_ops/).
  zerodce_baseline/   Zero-DCE + YOLOv11-n two-stage baseline: offline batch
                      enhancement, training/validation scripts, and the
                      official pre-trained DCE-Net weights (Epoch99.pth).
  jetson_bench/       Embedded latency benchmark scripts (NVIDIA Jetson).
weights/
  paper_final/        Final checkpoints of the paper models (seed 42):
                      Tiny-YOLO-EdgeFca_s.pt, Tiny-YOLO-EdgeFca_n.pt
  *.pt                Checkpoints of all ablation/seed runs, named by the
                      training run (prefix 5seed-/67- = random seed 5/67;
                      see the paper's Tables II and IV for the mapping).
dataset/              Download SLLI_dataset.zip from the GitHub Release
                      (SLLI v1.0) and unzip here.
```

## SLLI Dataset

The **Space Low-Light Image (SLLI)** dataset contains 14,449 images screened
from SPARK2022 via a two-stage procedure (quantitative statistical filtering +
manual visual refinement), covering 11 space-object categories, partitioned
into 10,114 / 2,167 / 2,168 images for train / val / test.

- Download: grab `SLLI_dataset.zip` (images + YOLO-format labels) from the
  **Release** page of this repository.
- Screening scripts: `code/LeYOLO-main/dataset_ops/` (`LLIscreen.py`,
  `LLIscreen_batch.py`).

## Training / Validation

All models were trained **from scratch** under a unified protocol
(SGD, 300 epochs, batch 32, 640x640, cosine LR, multi-scale, mixup 0.3).
See the paper (Sec. IV) for details.

```bash
cd code/LeYOLO-main
# example: train the small variant
python train.py
# validation on the SLLI test split
yolo val model=runs/detect/<run>/weights/best.pt data=SLLI.yaml split=test
```

## Citation

If you use this code or the SLLI dataset, please cite the paper.
