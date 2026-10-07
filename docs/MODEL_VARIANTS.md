# Model Variant Map

Mapping between config YAMLs, module combinations, and the paper's tables.
All structures below were verified by parsing the config files and the
`args.yaml` of each training run.

## 1. Proposed models — `code/LeYOLO-main/` (customized Ultralytics fork)

- Custom modules: `ultralytics/nn/modules/`
  - `Edge_fca.py` — EdgeFcaV2 (Laplacian-initialized edge extraction + FcaNet
    frequency attention + adaptive spatial gating); `Edge_fca_adp.py` holds
    component-ablation variants
  - `topkpooling.py` — TOPKSPP (MMA-SPPF, multi-maximum aggregation)
  - `fcanet.py`, `FEN.py`, `SCconv.py`, `smoothblock.py`, `penet.py`,
    `ii_block.py`, `gamma.py`, `HWB.py` — supporting blocks
- Model configs: `ultralytics/cfg/cfg/*.yaml`
- Dataset config: `ultralytics/cfg/datasets/SLLFS.yaml`
- Training entry: `train.py` / `main.py` (seed-controlled); training uses the
  standard Ultralytics trainer and `v8DetectionLoss` — no custom loss.

| Config (ultralytics/cfg/cfg/) | Module combination | Paper |
|---|---|---|
| `tiny_m_topkedge21.yaml` | MMA-SPPF + EdgeFcaV2 at P5 (C=96, d_h=20), P4 top-down (192, 80), P4 bottom-up (128, 40) | Tiny-YOLO-EdgeFca$_s$; Table II; Table IV full config |
| `tiny_s_topkedge21.yaml` | nano variant, 1.19 M params | Tiny-YOLO-EdgeFca$_n$; Table II |
| `Tiny_m_spp.yaml` | SPPF baseline (no EdgeFca) | Table IV row 1 |
| `Tiny_m_topk.yaml` | MMA-SPPF only | Table IV row 2 |
| `tiny_m_topkedge32.yaml` | MMA + EdgeFca (P5 only) | Table IV row 3 |
| `tiny_m_topkedge31.yaml` | MMA + EdgeFca (P5 + P4$_2$) | Table IV row 4 |
| `tiny_m_spp_edge21.yaml` | SPPF + EdgeFca (P5 + P4$_1$ + P4$_2$) | Table IV row 5 |
| `tiny_m_topkedge22.yaml` | MMA + EdgeFca (P5 + P3 + P4$_2$) | Table IV row 7 |
| `leyolonano_edge.yaml`, `leyolosmall_edge.yaml` | direct EdgeFca embedding | Table V |
| `Tinyolo_snod_*C2PSA*.yaml`, `Tinyolo_snod_*edge*.yaml` | module replacement / hybrid integration | Table V |
| `tiny_m_topkedge11/41/6.yaml`, others | exploratory variants (not reported in the paper) | — |

## 2. Baselines — `code/ultralytics-baselines/` (customized Ultralytics)

- Custom baseline architectures:
  - `ultralytics/cfg/models/v3/EMV-yolov3-tiny.yaml` + `nn/modules/EMV.py`
  - `ultralytics/cfg/models/v10/wtefnet.yaml` + `nn/modules/wtefnet.py`
  - `ultralytics/cfg/models/v3/yola.yaml`
- Standard configs used as-is: `v9/yolov9t`, `v10/yolov10n`, `11/yolo11n`,
  `12/yolo12n`, `rt-detr/rtdetr-r18`, `rt-detr/rtdetr-r34`
- Dataset config: `ultralytics/cfg/datasets/SLLFS.yaml`
- Training entry: `train.py` / `main.py` at the repository root.

## 3. Weights — `weights/`

- `paper_final/`: Tiny-YOLO-EdgeFca_s.pt, Tiny-YOLO-EdgeFca_n.pt (seed 42)
- Run-named `*.pt`: directory name of the training run; prefixes `5seed-`/`5-`
  and `67seed-`/`67-` denote random seeds 5 and 67; the remaining characters
  map to the config names above.
- Component ablations (`without edge.pt`, `without spatial gate.pt`,
  `without edge and gate.pt`) correspond to code-level variants of
  `Edge_fca.py` (see `Edge_fca_adp.py`).
