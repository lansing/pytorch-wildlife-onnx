# Dataset Curation & Evaluation

This document covers the datasets assembled for fine-tuning and evaluating MDV6-yolov10 models, the Make targets used to build and evaluate them, and expected download sizes.

All dataset targets run inside the `pytorch-wildlife-export-trt` Docker image so they share the same Python environment as the export pipeline.  The host `data/` directory is mounted at `/data` inside the container; annotation caches land in `cache/`.

---

## Training / Fine-tuning Dataset

The fine-tuning dataset (`data/md_ft/`) is assembled from two sources and used to train the QAT (Quantization-Aware Training) model.

### Sources

| Source | Content | Est. size |
|---|---|---|
| **WCS Camera Traps** | African and global wildlife — 2 500 animal-primary images + up to ~52 vehicle images. COCO Camera Traps JSON → YOLO labels. | ~3.5–5 GB |
| **COCO 2017** | 1 500 person-primary images + 500 vehicle images filtered from the standard COCO validation set. | ~300–500 MB |

The two sources are merged, shuffled, and split 80 / 10 / 10 into `train` / `val` / `test`.  The assembled layout lives under `data/md_ft/` and is described by `data/md_ft/megadetector_ft.yaml`.

### Make commands

```bash
# Small test build: 100 images per class (~300–500 MB)
make dataset-build-test

# Full production build: ~4–5 GB — confirm disk space first
make dataset-build
```

> **Budget warning:** the default production build downloads roughly 4–5 GB.  Raising `--wcs-max-animal` above 3 500 risks approaching 10 GB.

WCS annotations are cached in `cache/wcs/` (~28 MB uncompressed) and reused on subsequent builds.

---

## OOD Evaluation Dataset — Caltech Camera Traps (CCT20)

The CCT20 dataset is used exclusively for **out-of-distribution (OOD) validation** — it is never used in training.

### Why OOD validation matters

The fine-tuning dataset is drawn from WCS Camera Traps (African and global megafauna sites).  Evaluating accuracy gains only on in-distribution data cannot distinguish genuine model improvement from overfitting to the training distribution.  CCT20 provides a strong sanity check: it covers SW USA wildlife (coyote, mule deer, cottontail rabbit, California ground squirrel, skunk, …) photographed at entirely different sites with different camera hardware.  If the QAT-finetuned model outperforms the baseline on CCT20 as well, the accuracy improvement is real.

### Dataset details

- **Source:** [Caltech Camera Traps (CCT20)](https://lila.science/datasets/caltech-camera-traps) via LILA
- **Annotations:** `caltech_bboxes_20200316.json` — COCO Camera Traps format with `bbox: [x, y, w, h]` in pixel coordinates
- **Images:** Azure LILA mirror at `caltech-unzipped/cct_images/{uuid}.jpg` — no authentication required
- **Category mapping:** fine-grained species → `animal` (MD class 0); `person`/`people`/`human` terms → `person` (class 1); `empty`/`unknown` → skipped
- **Sample size:** 500 animal-primary images (default), reproducibly sampled with seed 42
- **Est. download:** ~200–350 MB (trail-cam JPEGs, typically 300–700 KB each)

### Make commands

```bash
# Download 500 CCT images (one-time, ~200–350 MB)
make dataset-download-cct-ood

# Override the sample size
make dataset-download-cct-ood CCT_MAX_ANIMAL=1000

# Evaluate a specific model against CCT OOD
make eval-ood MODEL=MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.engine
make eval-ood MODEL=MDV6-yolov10-c_int8_640_denorm_nhwc_uint8input.engine
```

The downloader writes `data/cct_ood/cct_ood.yaml` (eval-only, `val` split only).  You can also call the eval directly:

```bash
python -m PytorchWildlife_Export.dataset.eval \
    exported_models/MDV6-yolov10-c_qat_int8_640_denorm_nhwc_uint8input.engine \
    --dataset data/cct_ood/cct_ood.yaml \
    --split val
```

---

## Evaluating Models

### Single-model eval

```bash
# In-distribution val split (data/md_ft/)
make eval MODEL=MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.engine

# OOD eval (data/cct_ood/)
make eval-ood MODEL=MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.engine
```

`CONF` defaults to `0.1`.  Both targets accept `.engine` and `.onnx` files from `exported_models/`.

### Sweep eval — evaluate everything at once

```bash
make sweep-eval
```

`sweep-eval` iterates over all standard MDV6-yolov10 model variants present in `exported_models/` and writes a CSV summary to `exported_models/eval_results_<split>.csv`.  Useful after a `make sweep-export` run.

Optional overrides:

```bash
# Evaluate only INT8 TensorRT engines on the test split
make sweep-eval SWEEP_EVAL_SPLIT=test SWEEP_EVAL_FORMATS=int8 SWEEP_EVAL_RUNTIMES=tensorrt

# Print full per-model eval tables (verbose)
make sweep-eval SWEEP_EVAL_VERBOSE=--verbose
```

---

## Directory Layout

```
data/
  md_ft/                    ← fine-tuning dataset (WCS + COCO)
    images/
      train/  val/  test/
    labels/
      train/  val/  test/
    megadetector_ft.yaml
  cct_ood/                  ← OOD evaluation dataset (CCT20)
    images/val/
    labels/val/
    cct_ood.yaml

cache/
  wcs/                      ← WCS annotation cache (~28 MB, reused across builds)
  cct/                      ← CCT annotation cache (~35 MB, downloaded once)

exported_models/            ← TRT engines / ONNX files
```
