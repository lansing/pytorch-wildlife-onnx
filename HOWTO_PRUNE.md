# Structured Channel Pruning — End-to-End Guide

This document explains how to prune a MegaDetector model, fine-tune it to
recover accuracy, export it for inference, and compare its metrics against
the baseline using pre-recorded eval results.

---

## Table of Contents

1. [What pruning does and why](#1-what-pruning-does-and-why)
2. [How it works technically](#2-how-it-works-technically)
3. [Targeting a different base model](#3-targeting-a-different-base-model)
4. [Running the pipeline end-to-end](#4-running-the-pipeline-end-to-end)
5. [Evaluating and recording results](#5-evaluating-and-recording-results)
6. [Comparing against the baseline](#6-comparing-against-the-baseline)
7. [Tuning knobs](#7-tuning-knobs)
8. [Combining pruning with QAT](#8-combining-pruning-with-qat)

---

## 1. What pruning does and why

**Structured channel pruning** permanently removes entire output channels
(filters) from Conv layers, reducing both model size and FLOPs.  Unlike weight
sparsity (zeroing individual weights), removing whole channels produces dense
weight matrices that execute faster on real hardware without any sparse-math
support.

The goal is a smaller model that still meets accuracy requirements.  In our
pipeline a pruned float16 TRT engine runs faster than the original while
occupying less memory, making it suitable for edge deployments with tight
latency or memory budgets.

**What we prune**: the _hidden channel count_ inside each C2f block.  C2f
(Cross-Stage Partial Feature Fusion) is the main computational unit in the
YOLOv10 backbone and neck.  Its internal width (denoted `c` in the code) is
independent of the block's external input/output interface, so we can shrink
it without touching any adjacent layers.

**What we don't prune**: standalone Conv layers between C2f blocks, the SPPF
pooling block, and the v10Detect head.  Pruning those would require
cross-layer dependency tracking that is tricky given YOLOv10's dynamic routing
structure.

---

## 2. How it works technically

### 2.1 Why torch-pruning doesn't work here

The standard tool for structured pruning is [torch-pruning](https://github.com/VainF/Torch-Pruning),
which builds a dependency graph via `torch.fx.symbolic_trace` and groups
coupled channels (e.g. channels shared across a skip connection or concat).

YOLOv10's `DetectionModel._forward_once()` routes activations with Python list
indexing (`y[m.f]`), which is not symbolically traceable by FX.  The tracer
finds zero prunable groups and exits silently without modifying anything.

### 2.2 C2f internal pruning (our approach)

`PytorchWildlife_Export/finetune/prune_train.py::prune_model()` implements
magnitude-based channel pruning directly on the `C2f` module's internals:

```
Input (c1 ch)
    │
   cv1 ──────────────────────────────────────────────────────────┐
   (c1 → 2·c)                                                    │
    │                                                            │
  chunk(2)                                                       │
    ├── chunk0 (c ch) ──────────────────────────────────────────>│ cat
    └── chunk1 (c ch)                                            │
         │                                                       │
       bn[0]    (c → c)                                         │
         │                                                       │
       bn[1]    (c → c)                                         │
         │                                                       │
        ...                                                     │
         │                                                      │
       bn[n-1]  (c → c) ───────────────────────────────────────>│
                                                                 │
                                                   cat((2+n)·c) │
                                                                 │
                                                       cv2 ──────┘
                                                   ((2+n)·c → c2)
                                                        │
                                                   Output (c2 ch)
```

To reduce `c → new_c` (where `new_c = round_to_divisor(c · (1 − p))`):

1. **Score channels**: compute L2 norm of each output channel's weight tensor
   in `cv1`.  The first-half channels (chunk0, pass-through) and second-half
   channels (chunk1, fed to bottlenecks) are scored independently.

2. **Select keepers**: `first_keep` = top-`new_c` indices from chunk0;
   `second_keep` = top-`new_c` indices from chunk1.

3. **Prune `cv1`**: replace its `nn.Conv2d` and `nn.BatchNorm2d` with new
   modules sliced to `[cv1_out_keep]` output channels.

4. **Prune each Bottleneck** in `c2f.m`:
   - `bn.cv1` input goes from `c` → `new_c` (input channels = `second_keep`
     for the first bottleneck; previous bottleneck's kept output channels for
     subsequent ones).
   - `bn.cv1` output is pruned to `new_bn_c` channels by importance.
   - `bn.cv2` input = `bn_cv1_out_keep`; output pruned to `new_c` channels by
     importance.

5. **Prune `cv2`**: its input is the concatenation of all segment outputs.
   We compute `cv2_in_keep` by offsetting each segment's kept indices into
   the original `(2+n)·c`-wide concatenation.

6. **Update `c2f.c`** to `new_c` so the block's bookkeeping is consistent.

CIB-style bottlenecks (`C2fCIB`, which uses `nn.Sequential` internally) are
skipped since they have a different structure.

### 2.3 FLOPs target vs actual reduction

The pruning ratio is computed from the FLOPs target:

```
(1 − p)² ≈ flops_fraction    →    p = 1 − √(flops_fraction)
```

Because C2f FLOPs scale as `c²`, pruning the hidden width by fraction `p`
reduces C2f FLOPs roughly by `(1−p)²`.  With `flops_fraction=0.50`, this
gives `p ≈ 0.29`.  Only C2f blocks are pruned, so the actual whole-model
FLOPs reduction is smaller (~23% on yolov10-c at 640px).

### 2.4 Fine-tuning

After pruning the model's forward pass is valid but accuracy drops because
the kept channels were chosen by weight magnitude, not by their contribution
to the output loss.  Fine-tuning on the training set restores accuracy:

- Same dataloader and loss function as QAT fine-tuning (`train_one_epoch`)
- Lower learning rate (5e-5 → 1e-6 cosine) to avoid disrupting the pruned
  structure
- 60 epochs recommended; 30 gives a useful rough result

### 2.5 Export

The pruned PyTorch model is exported to ONNX via `torch.onnx.export` with
`export=True` on the detect head (produces NMS-ready `(1, 300, 6)` output).
This ONNX is then passed to `export_tool.py` via `--onnx-override`, which
prepends the uint8/nhwc/denorm preprocessing graph and compiles to a TRT
engine — exactly the same final stage as for unmodified models.

---

## 3. Targeting a different base model

### 3.1 Create a config file

Copy `config_prune_c640.yaml` and edit it:

```bash
cp config_prune_c640.yaml config_prune_e640.yaml
```

Key fields to change:

```yaml
model:
  version: "MDV6-yolov10-e"   # ← model hub version string
  weights: null                # null = hub download; or path to a local .pt

data:
  dataset_yaml: "/data/md_ft/megadetector_ft.yaml"
  imgsz: 640
  batch_size: 4                # ← reduce for larger models if OOM

prune:
  flops_fraction: 0.50         # keep this fraction of FLOPs; 0.50 = 2× reduction
  channel_divisor: 8           # keep channel counts divisible by 8 (Tensor Core)

train:
  epochs: 60

output:
  checkpoint_dir: "/app/checkpoints/prune_e640"
  onnx_path: "/app/checkpoints/prune_e640/MDV6-yolov10-e_pruned.onnx"
```

### 3.2 Add Makefile variables for the new model

At the top of your `make train-prune` invocation (or export it in your shell):

```bash
# Fine-tune
make train-prune \
  PRUNE_CONFIG=config_prune_e640.yaml \
  PRUNE_EPOCHS=60

# Export the pruned ONNX to a TRT engine
make export-pruned \
  PRUNED_ONNX=/app/checkpoints/prune_e640/MDV6-yolov10-e_pruned.onnx \
  PRUNED_MODEL_VERSION=MDV6-yolov10-e \
  PRUNED_OUTPUT=/exported_models/MDV6-yolov10-e_pruned_float16_640_denorm_nhwc_uint8input.engine

# Evaluate
make eval-pruned \
  PRUNED_ENGINE=MDV6-yolov10-e_pruned_float16_640_denorm_nhwc_uint8input.engine
```

Or override `PRUNED_*` variables permanently by adding them to your
`GNUmakefile.local` (if you have one) or passing them every time.

---

## 4. Running the pipeline end-to-end

### Step 0 — Prerequisites

```bash
make dataset-build          # build the fine-tuning dataset (WCS + COCO, ~4 GB)
make dataset-download-cct-ood  # download OOD validation images (~300 MB)
```

### Step 1 — Prune and fine-tune

```bash
make train-prune                     # 30 epochs (quick run)
make train-prune PRUNE_EPOCHS=60     # 60 epochs (recommended)
```

This runs inside the `pytorch-wildlife-export-trt` Docker container.  What
happens:

1. `pip install torch-pruning` (takes ~30 s the first run, cached thereafter)
2. The base model is downloaded from the hub (or loaded from `--weights`)
3. Data loaders are built from the fine-tuning dataset
4. `prune_model()` is called — see §2 for details
5. 30/60-epoch fine-tuning loop runs, saving checkpoints each epoch
6. After each epoch the best-val-loss checkpoint is re-exported to ONNX so
   you always have a usable artifact even if you kill the job early

Checkpoints are written to `checkpoints/prune_c640/` (or the dir in your
config).  The best-val-loss ONNX is at
`checkpoints/prune_c640/MDV6-yolov10-c_pruned.onnx`.

**Resume from a checkpoint** (e.g. if the job was killed at epoch 15):

```bash
make train-prune PRUNE_EPOCHS=60 PRUNE_SKIP=--resume
```

The `--resume` flag tells `prune_train.py` to load the latest checkpoint from
`checkpoint_dir` and continue training from there without re-pruning.

### Step 2 — Export to TRT

```bash
make export-pruned
```

This calls `export_tool.py --onnx-override` with the pruned ONNX.  The export
tool skips the PyTorch→ONNX step and goes straight to:

1. Load the pruned ONNX
2. Prepend preprocessing nodes (uint8 cast → /255 → NHWC→NCHW transpose)
3. Compile to a TRT float16 engine

The engine is written to
`exported_models/MDV6-yolov10-c_pruned_float16_640_denorm_nhwc_uint8input.engine`.

### Step 3 — Evaluate

```bash
make eval-pruned
```

Runs two eval passes and appends one row each to:

- `exported_models/eval_results_val.csv` — in-distribution (our val split)
- `exported_models/eval_results_ood.csv` — OOD (Caltech Camera Traps)

---

## 5. Evaluating and recording results

### One-off eval with CSV output

Any model (pruned or otherwise) can be evaluated and its results saved:

```bash
# In-distribution
make eval \
  MODEL=MDV6-yolov10-c_pruned_float16_640_denorm_nhwc_uint8input.engine \
  CSV=/exported_models/eval_results_val.csv

# OOD
docker run --runtime nvidia ... \
  -m PytorchWildlife_Export.dataset.eval \
  /exported_models/MDV6-yolov10-c_pruned_float16_640_denorm_nhwc_uint8input.engine \
  --dataset /data/cct_ood/cct_ood.yaml \
  --split val --conf 0.1 \
  --csv /exported_models/eval_results_ood.csv
```

The `--csv` flag appends one row in the standard format.  The file is created
with a header if it doesn't exist yet; otherwise the row is appended.

### CSV row format

```
model,model_version,format,size,runtime,n_images,
animal_AP50,animal_AP50_95,animal_AR50,animal_AR50_95,
person_AP50,person_AP50_95,person_AR50,person_AR50_95,
vehicle_AP50,vehicle_AP50_95,vehicle_AR50,vehicle_AR50_95,
mAP50,mAP50_95,mAR50,mAR50_95
```

`model_version` and the other metadata columns are parsed automatically from
the model filename.  The naming convention
`{version}_{format}_{size}_{flags}.{ext}` is expected:

```
MDV6-yolov10-c_pruned_float16_640_denorm_nhwc_uint8input.engine
 └── version ─┘         └─format┘ └size┘                 └─ runtime
```

---

## 6. Comparing against the baseline

### 6.1 Using pre-recorded results

`exported_models/eval_results_val.csv` contains eval results for all
previously exported models.  There is no need to re-run them.  Just load the
CSV and filter:

```python
import pandas as pd

df = pd.read_csv("exported_models/eval_results_val.csv")

# Compare pruned vs original MDV6-yolov10-c at 640px float16 TRT
rows = df[
    df["model_version"].isin(["MDV6-yolov10-c", "MDV6-yolov10-c_pruned"]) &
    (df["size"] == 640) &
    (df["format"] == "float16") &
    (df["runtime"] == "tensorrt")
][["model_version", "animal_AP50", "mAP50", "mAR50"]].set_index("model_version")

print(rows)
```

### 6.2 Quick shell comparison

```bash
python3 - <<'EOF'
import pandas as pd, sys

val = pd.read_csv("exported_models/eval_results_val.csv")
ood = pd.read_csv("exported_models/eval_results_ood.csv")

COLS = ["model", "animal_AP50", "person_AP50", "vehicle_AP50", "mAP50"]

print("=== In-distribution (val) ===")
print(val[COLS].to_string(index=False))

print("\n=== OOD (CCT) ===")
print(ood[COLS].to_string(index=False))
EOF
```

### 6.3 Example comparison table

After running `make eval-pruned` for `MDV6-yolov10-c` at 640px, the tables
look like this (30-epoch fine-tune):

**In-distribution (val split, 459 images)**

| model_version           | animal AP50 | mAP50  | FLOPs |
|-------------------------|-------------|--------|-------|
| MDV6-yolov10-c          | 0.8219      | 0.4569 | 4.16G |
| MDV6-yolov10-c_pruned   | 0.7373      | 0.4647 | 3.20G |
| Δ                       | −8.5 pp     | +0.8pp | −23%  |

**OOD (Caltech Camera Traps, 541 images)**

| model_version           | animal AP50 | mAP50  |
|-------------------------|-------------|--------|
| MDV6-yolov10-c          | —           | —      |
| MDV6-yolov10-c_pruned   | 0.5904      | 0.1968 |

> OOD baseline not yet in the CSV — run `make eval MODEL=MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.engine CSV=/exported_models/eval_results_ood.csv` once to add it, then the comparison is fully automated.

---

## 7. Tuning knobs

| Parameter | Config key / Makefile var | Default | Notes |
|-----------|--------------------------|---------|-------|
| Base model | `model.version` | `MDV6-yolov10-c` | Any MDV6 hub version |
| Local weights | `model.weights` | `null` | Skip hub download |
| FLOPs target | `prune.flops_fraction` | `0.50` | 0.65 for lighter pruning |
| Channel divisor | `prune.channel_divisor` | `8` | Tensor Core alignment |
| Fine-tune epochs | `train.epochs` / `PRUNE_EPOCHS` | `60` | 30 for quick experiments |
| Learning rate | `train.lr` | `5e-5` | Lower if loss diverges |
| Batch size | `data.batch_size` | `8` | Reduce if OOM |
| Input size | `data.imgsz` | `640` | Must match export size |
| Export format | `PRUNED_FORMAT` | `float16` | `int8` also works (see §8) |

**Choosing `flops_fraction`**:

- `0.50` → p ≈ 0.29 → ~23% total FLOPs reduction (modest, recommended start)
- `0.35` → p ≈ 0.41 → ~30% total FLOPs reduction (more aggressive)
- `0.25` → p ≈ 0.50 → ~35% total FLOPs reduction (high drop, needs more epochs)

The actual reduction is less than the C2f-theoretical target because
standalone Conv layers between C2f blocks are not pruned.

---

## 8. Combining pruning with QAT

To get maximum compression (fewer channels + INT8 weights):

```
prune → fine-tune → QAT fine-tune → extract scales → INT8 export
```

### Step 1 — Prune and fine-tune (as above)

```bash
make train-prune PRUNE_EPOCHS=60
```

### Step 2 — Load the pruned checkpoint into QAT

```bash
make train-qat \
  QAT_WEIGHTS=checkpoints/prune_c640/prune_MDV6-yolov10-c_epoch060.pt
```

`qat_train.py` accepts `--weights` pointing to a pruned checkpoint saved by
`prune_train.py`.  It loads the full module with `weights_only=False` so the
pruned channel structure is preserved.

### Step 3 — Extract learned scales

```bash
python3 -m PytorchWildlife_Export.finetune.qat_train \
  --config config_qat_c640.yaml \
  --extract-scales \
  --output-scales checkpoints/prune_c640/qat_scales.json
```

### Step 4 — Export pruned + INT8

```bash
make export-pruned \
  PRUNED_FORMAT=int8 \
  PRUNED_ONNX=/app/checkpoints/prune_c640/MDV6-yolov10-c_pruned.onnx \
  PRUNED_OUTPUT=/exported_models/MDV6-yolov10-c_pruned_int8_640_denorm_nhwc_uint8input.engine
```

Add `--scales_json` to the `export-pruned` target if you want QAT-learned
scales to override ORT calibration.

### Step 5 — Evaluate

```bash
make eval-pruned \
  PRUNED_ENGINE=MDV6-yolov10-c_pruned_int8_640_denorm_nhwc_uint8input.engine
```

Results are appended to the standard CSVs for comparison.
