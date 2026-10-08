# Reproduction Guide

This guide reproduces the main results of the paper. All experiments share one
environment and one training protocol; only the model YAML and the random seed
change between runs.

## 0. Environment and dataset

```bash
conda env create -f environments/torch310_environment.yml
conda activate torch310          # Python 3.10, torch 2.4.0+cu118, ultralytics 8.3.49
```

Download `SLLI_dataset.zip` from the GitHub Release (SLLI-v1.0), unzip it, and
edit the `path:` field in
`code/<repo>/ultralytics/cfg/datasets/SLLFS.yaml` to point to the unzipped
`SLLFS/` directory (contains `images/{train,val,test}` + `labels/`).
Splits: 10,114 / 2,167 / 2,168 images, 11 classes.

## 1. Unified training protocol (all models, all tables)

Identical for every model in Table II / IV / V (verified against the training
logs of every run):

```bash
yolo detect train \
  model=<model.yaml> \
  data=ultralytics/cfg/datasets/SLLFS.yaml \
  epochs=300 batch=32 imgsz=640 \
  optimizer=SGD lr0=0.01 momentum=0.937 cos_lr=True \
  multi_scale=True warmup_epochs=5 mixup=0.3 \
  seed=<42|5|67> \
  project=runs/detect name=<run_name>
```

No pretrained weights are loaded (from-scratch, YAML-only initialization), and
no per-model tuning is applied. Set the seed inside `train.py` / `main.py` if
you prefer the script entry (both set `torch.backends.cudnn.deterministic`).

Validation on the test split:

```bash
yolo val model=runs/detect/<run_name>/weights/best.pt \
  data=ultralytics/cfg/datasets/SLLFS.yaml split=test
```

## 2. What to train for each table

| Paper result | Repository | model YAML (under `ultralytics/cfg/`) | Seeds |
|---|---|---|---|
| Table II, Tiny-YOLO-EdgeFca_s | LeYOLO-main | `cfg/tiny_m_topkedge21.yaml` | 42, 5, 67 |
| Table II, Tiny-YOLO-EdgeFca_n | LeYOLO-main | `cfg/tiny_s_topkedge21.yaml` | 42, 5, 67 |
| Table IV row 1–7 | LeYOLO-main | see `docs/MODEL_VARIANTS.md` | 42, 5, 67 |
| Table II YOLO/RT-DETR rows | ultralytics-baselines | `models/v9/yolov9t.yaml`, `models/v10/yolov10n.yaml`, `models/11/yolo11n.yaml`, `models/12/yolo12n.yaml`, `models/rt-detr/rtdetr-{r18,r34}.yaml` | 42, 5, 67 |
| Table II EMV/WTEFNet/YOLA | ultralytics-baselines | `models/v3/EMV-yolov3-tiny.yaml`, `models/v10/wtefnet.yaml`, `models/v3/yola.yaml` | 42, 5, 67 |
| Table V integrations | LeYOLO-main | `cfg/leyolonano_edge.yaml`, `cfg/leyolosmall_edge.yaml`, `cfg/Tinyolo_snod_*.yaml` | 42 (single-seed exploratory) |

## 3. Expected results (mean ± std over seeds 42/5/67)

| Model | mAP50 | mAP75 | mAP90 | mAP50-95 |
|---|---|---|---|---|
| Tiny-YOLO-EdgeFca_n | 0.791 ± 0.019 | 0.609 ± 0.013 | 0.174 ± 0.023 | 0.541 ± 0.010 |
| Tiny-YOLO-EdgeFca_s | 0.799 ± 0.038 | 0.653 ± 0.016 | 0.205 ± 0.009 | 0.568 ± 0.014 |
| RT-DETR-R18 | 0.833 ± 0.011 | 0.679 ± 0.015 | 0.167 ± 0.021 | 0.583 ± 0.006 |
| SPPF baseline (Table IV) | 0.803 ± 0.011 | 0.626 ± 0.011 | 0.175 ± 0.015 | 0.551 ± 0.009 |

Exact per-seed values are reported in the paper's reply material; small
deviations (~±0.01) are expected from GPU nondeterminism.

## 4. Zero-DCE + YOLOv11-n two-stage baseline

```bash
cd code/zerodce_baseline
# (a) offline enhancement of all SLLI images with the official pre-trained DCE-Net
python zerodce_batch_gpu.py          # writes the enhanced dataset copy
# (b) train YOLOv11-n from scratch on the enhanced images (same protocol as above)
python train_zerodce_yolo11n.py      # seed 42 / 5 / 67 variants provided
```

Expected: mAP50 0.776 ± 0.011, mAP50-95 0.501 ± 0.003.

## 5. Gating-map visualization (Fig. of Sec. III)

```bash
cd code/LeYOLO-main
python g_vis_all_layers.py    # E/G maps at all three EdgeFca positions
```

`gate_collapse.py` and `per_sample_g.py` reproduce the seed-variance
diagnostics of the reply letter.

## 6. SLLI dataset construction (optional)

The two-stage screening pipeline from SPARK2022 is in
`code/LeYOLO-main/dataset_ops/` (`LLIscreen.py`, `LLIscreen_batch.py`).
Rebuilding is unnecessary if you use the released SLLI zip directly.
