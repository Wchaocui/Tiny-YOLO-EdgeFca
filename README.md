# Tiny-YOLO-EdgeFca

Code, pre-trained weights, and the **SLLI** dataset for
**"Tiny-YOLO-EdgeFca: A Lightweight Framework for Low-Light Space Object Detection"**.

> Code, weights, and the SLLI dataset are publicly released to support
> reproducibility.

## Repository Structure

```
code/
  LeYOLO-main/        Custom Ultralytics fork containing the proposed modules
                      (EdgeFca, MMA-SPPF/TOPKSPP, lightweight backbone) and
                      the SLLI screening scripts (dataset_ops/).
  ultralytics-baselines/  Customized Ultralytics used to train the compared
                      baselines, including the custom EMV-YOLOv3-tiny,
                      WTEFNet, and YOLA architectures (config + module code).
  zerodce_baseline/   Zero-DCE + YOLOv11-n two-stage baseline: offline batch
                      enhancement, training/validation scripts, and the
                      official pre-trained DCE-Net weights (Epoch99.pth).
  jetson_bench/       Embedded latency benchmark scripts (NVIDIA Jetson).
docs/
  MODEL_VARIANTS.md   Run <-> config-YAML <-> module-combination map for every
                      model variant and its location in the paper's tables.
  REPRODUCTION.md     Step-by-step reproduction guide: environment, dataset,
                      unified training protocol, exact commands per table,
                      and expected mean +/- std results.
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

## Environment

The reference conda environment used for all training/validation experiments
is provided in `environments/torch310_environment.yml`
(Python 3.10, PyTorch 2.4.0+cu118, Ultralytics 8.3.49):

```bash
conda env create -f environments/torch310_environment.yml
conda activate torch310
```

`environments/torch310_environment_legacy.yml` is an earlier export kept for
reference.

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
