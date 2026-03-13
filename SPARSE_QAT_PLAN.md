# Sparse QAT Fine-Tuning Plan — MDV6-yolov10-e

**Purpose**: Recover mAP after 2:4 magnitude pruning, then rebuild the INT8 TRT engine with
jointly-optimized sparse weights and quantization scales.
**Prerequisites**: Ampere sparsity test on target hardware shows meaningful throughput gain
over dense INT8 baseline before committing to this effort.

---

## Contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Training data strategy](#2-training-data-strategy)
3. [Fine-tuning infrastructure](#3-fine-tuning-infrastructure)
4. [Phase A: Sparse fine-tuning (accuracy recovery)](#4-phase-a-sparse-fine-tuning)
5. [Phase B: Sparse QAT fine-tuning](#5-phase-b-sparse-qat)
6. [Export and engine rebuild](#6-export-and-engine-rebuild)
7. [Evaluation checkpoints](#7-evaluation-checkpoints)
8. [Optimization deep dive](#8-optimization-deep-dive)

---

## 1. Pipeline overview

```
Dense pretrained .pt
        │
        ▼
2:4 magnitude pruning          ← already implemented in quant.py
(apply_2_4_sparsity_to_model)   (applied at export time to the ONNX graph)
        │
        ▼
Sparse fine-tuning             ← Phase A (Ultralytics model.train())
(restore mAP, mask held fixed)  (~30 epochs, ~3-5k examples/class)
        │
        ▼
Sparse QAT fine-tuning         ← Phase B (NVIDIA ModelOpt mtq.quantize + train loop)
(recover INT8 accuracy,         (~30 more epochs, same dataset)
 sparsity + scales co-optimised)
        │
        ▼
ONNX export with QDQ nodes     ← torch.onnx.export() on the QAT model
(replaces our current PTQ path)
        │
        ▼
TRT engine build               ← existing trt_export.py unchanged
(INT8 explicit + SPARSE_WEIGHTS flag)
```

**Key difference from current pipeline**: Today we apply pruning *after* ONNX export
(post-training, inside `yolo_exporter.py`).  For sparse QAT the sparsity mask must be
baked into the PyTorch model *before* training, and held frozen while the quantization
scales and weights adapt during fine-tuning.  The ONNX export comes at the very end.

---

## 2. Training data strategy

### Class targets

| Class   | Labeled instances (min) | Labeled instances (target) |
|---------|------------------------|---------------------------|
| animal  | 2,000                  | 5,000                     |
| person  | 1,000                  | 2,000                     |
| vehicle | 500                    | 1,000                     |

These are instance counts in the *training* split, not image counts.  One image may
contain multiple instances, so effective image count will be lower.

### Data sources (minimal download)

**Primary: LILA WCS Camera Traps** (animals + vehicles — camera-trap style, closest to
MDV6 training distribution)

- URL: `https://lila.science/datasets/wcscameratraps`
- Cloud paths:
  - AWS: `s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs-unzipped/`
  - GCP: `gs://public-datasets-lila/wcs-camera-traps/`
- Format: COCO Camera Traps JSON (`wcs_20220205_bboxes.json`)
- Annotation keys used: `"category_id"` (maps to MDV6 classes: 1=animal, 2=person, 3=vehicle)
- Access strategy: download annotations JSON first (~10 MB), build a list of image URLs per
  class, then pull only the specific images needed — no full zip required.

```bash
# Step 1: get annotation index (~10 MB)
aws s3 cp \
  s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs-unzipped/wcs_20220205_bboxes.json \
  data/wcs_bboxes.json

# Step 2: run our sampling script (see §3) to select images, convert to YOLO format,
#         and download only the selected image files
python tools/prepare_sparse_qat_data.py \
  --annotation data/wcs_bboxes.json \
  --base_url   s3://us-west-2.opendata.source.coop/agentmorris/lila-wildlife/wcs-unzipped \
  --out_dir    data/sparse_qat \
  --n_animal   5000 \
  --n_person   2000 \
  --n_vehicle  1000
```

**Gap-filler: COCO 2017 (person + vehicle)**

WCS excludes people for privacy; NACTI has limited vehicle coverage.  COCO 2017 fills the
gap.  Use the official COCO API or `fiftyone` to download only images containing the
target classes.

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Download at most 2000 COCO images containing a person or vehicle
dataset = foz.load_zoo_dataset(
    "coco-2017", split="train",
    label_types=["detections"],
    classes=["person", "car", "truck", "motorcycle"],
    max_samples=2000,
)
# Export to YOLO format, remapping COCO class names to {person→1, vehicle→2}
```

**Validation set**: Use the WCS or COCO *val* splits (same selective download).  Target
~500 images per class for a statistically meaningful mAP estimate during training.

---

## 3. Fine-tuning infrastructure

### Option A: CameraTraps/PW_FT_detection (recommended for Phase A)

The `CameraTraps/PW_FT_detection/main.py` wrapper is a thin Ultralytics trainer driven by
`config.yaml`.  It supports YOLO and RTDETR models, resume, and experiment naming.

**Limitations** (need minor patches):
- `freeze` parameter is not exposed — add it to `model.train()` call in `main.py`
- No support for sparse-mask preservation — use a custom optimizer hook (see §4)

Patch to add `freeze` (minimal change):

```python
# CameraTraps/PW_FT_detection/main.py  (model.train call, ~line 28)
results = model.train(
    data=cfg.data,
    epochs=cfg.epochs,
    imgsz=cfg.imgsz,
    device=cfg.device_train,
    save_period=cfg.save_period,
    workers=cfg.workers,
    batch=cfg.batch_size_train,
    val=cfg.val,
    project=f"runs/train_{cfg.exp_name}",
    name="exp",
    patience=cfg.patience,
    resume=cfg.resume,
    freeze=getattr(cfg, "freeze", None),     # add this line
    lr0=getattr(cfg, "lr0", 0.01),           # expose lr0
    cos_lr=getattr(cfg, "cos_lr", False),    # expose cosine LR
)
```

### Option B: Custom PyTorch loop (required for Phase B QAT)

Phase B uses ModelOpt's `mtq.quantize()` which wraps the model in fake-quantization
modules.  Ultralytics' `model.train()` API does not support this.  A standalone PyTorch
training loop is needed.  ModelOpt provides `modelopt.torch.utils.train_utils` helpers for
this.

### Data format

Standard YOLO txt format:
```
<class_idx> <x_center> <y_center> <width> <height>   (normalized to [0,1])
```

Class mapping (must match MDV6 training):
```yaml
names:
  0: animal
  1: person
  2: vehicle
```

Data YAML (`data/sparse_qat/megadetector_ft.yaml`):
```yaml
path: /abs/path/to/data/sparse_qat
train: images/train
val:   images/val
nc: 3
names:
  0: animal
  1: person
  2: vehicle
```

---

## 4. Phase A: Sparse fine-tuning

**Goal**: Recover mAP after 2:4 pruning.  No quantization in this phase.
**Input**: `MDV6-yolov10-e.pt` (dense original) + 2:4 sparsity mask applied to weights.

### Step 1 — Apply sparsity to the PyTorch model

```python
# tools/apply_sparsity_to_pt.py
import torch
from PytorchWildlife_Export.model_exporters.quant import apply_2_4_sparsity_to_model
import onnx

# Load the Ultralytics model
from ultralytics import YOLO
model = YOLO("checkpoints/MDV6-yolov10-e-1280.pt")

# We need the raw nn.Module to apply sparsity to Conv weight tensors.
# apply_2_4_sparsity_to_model operates on ONNX; for the .pt model we need
# an equivalent PyTorch-native version (implement below).
```

**Note**: Our current `apply_2_4_sparsity_to_model` operates on ONNX initializers.
For Phase A we need a PyTorch-native variant that directly modifies `nn.Conv2d.weight`
tensors in the loaded model.  This is a small new utility to write:

```python
# PytorchWildlife_Export/tools/apply_sparsity_pt.py

import torch
import torch.nn as nn
import numpy as np
from PytorchWildlife_Export.model_exporters.quant import _apply_2_4_sparsity

def apply_2_4_sparsity_to_pytorch_model(model: nn.Module, exclude_prefixes=("model.0", "model.23")):
    """Apply 2:4 magnitude sparsity to Conv2d weight tensors in-place.
    Returns (model, stats_dict).
    """
    stats = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        w = module.weight.data.cpu().numpy()
        w_sparse = _apply_2_4_sparsity(w)
        module.weight.data = torch.from_numpy(w_sparse).to(module.weight.device)
        total = w.size
        zeroed = int((w_sparse == 0).sum()) - int((w == 0).sum())
        stats[name] = {"total": total, "newly_zeroed": zeroed}
    return model, stats
```

### Step 2 — Freeze the sparsity mask during training

After applying sparsity, we need to re-zero the pruned positions after every optimizer step
so that gradient updates cannot fill in the zeroed weights:

```python
# Mask hook — attach after model creation, before training loop
def build_sparsity_masks(model, exclude_prefixes):
    """Return {param_name: bool_mask} for Conv2d weights."""
    masks = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        masks[name + ".weight"] = (module.weight.data != 0)
    return masks

def apply_masks(model, masks):
    """Call after optimizer.step() to re-zero pruned positions."""
    for name, param in model.named_parameters():
        if name in masks:
            param.data.mul_(masks[name].float())
```

Ultralytics does not expose a callback for this, so for Phase A use the `PW_FT_detection`
wrapper with a monkey-patched optimizer step, or switch to the custom loop from §3-B.

### Step 3 — Training configuration

```yaml
# CameraTraps/PW_FT_detection/config_sparse_ft.yaml
model:      YOLO
model_name: MDV6-yolov10-e       # loaded from sparse checkpoint instead
data:       /path/to/megadetector_ft.yaml
task:       train
exp_name:   sparse_ft_phase_a

epochs:         30               # ~10% of original training schedule
batch_size_train: 16
imgsz:          640
device_train:   0
workers:        8
lr0:            0.0001           # 100x lower than default; fine-tuning LR
cos_lr:         true
freeze:         10               # freeze first 10 backbone layers
patience:       10
save_period:    5
val:            true
resume:         false
```

**Expected outcome**: mAP@50 within ~1-2 points of dense baseline after 30 epochs.

---

## 5. Phase B: Sparse QAT

**Goal**: Co-optimize quantization scales and weights together, with sparsity mask frozen.
**Input**: Phase A checkpoint (sparse .pt, mAP recovered).
**Tool**: NVIDIA ModelOpt (`pip install nvidia-modelopt[torch]`).

### Step 1 — Install ModelOpt

```bash
pip install "nvidia-modelopt[torch]>=0.21"
# Verify:
python -c "import modelopt; print(modelopt.__version__)"
```

### Step 2 — Insert fake-quantization (PTQ calibration pass)

```python
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
import copy

# Load sparse Phase-A checkpoint
from ultralytics import YOLO
ul_model = YOLO("runs/train_sparse_ft_phase_a/exp/weights/best.pt")
pt_model  = ul_model.model.model   # raw nn.Module

# Build selective quantization config — mirrors our existing PTQ exclusions.
# model.0:  3-channel input, no INT8 Tensor Core path on Turing/Ampere
# model.23: detection head — start excluded, optionally re-enable after initial calibration
quant_cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
quant_cfg["quant_cfg"]["*model.0*"]  = {"enable": False}
quant_cfg["quant_cfg"]["*model.23*"] = {"enable": False}

# Calibration forward loop (~100 images from training set, no labels needed)
def calibrate(model):
    model.eval()
    with torch.no_grad():
        for imgs, _ in calibration_loader:   # DataLoader over 100 cal images
            model(imgs.cuda())

pt_model = mtq.quantize(pt_model, quant_cfg, calibrate)

# Inspect which layers got Q/DQ and their exact names:
mtq.print_quant_summary(pt_model)

# Save calibrated checkpoint
mto.save(pt_model, "checkpoints/sparse_ptq_calibrated.pth")
```

### Step 3 — Optional: re-enable head quantization before QAT

If QAT should also tune the head (model.23) into INT8:

```python
# After calibration save, before QAT loop:
mtq.enable_quantizer(pt_model, "*model.23*")
# The head's quantizers will be initialized with a safe default scale;
# QAT fine-tuning will adapt both scales and weights for the head as well.
```

This is worth trying: the head was excluded from PTQ because calibration-only scales
were insufficiently accurate, but with gradient-based QAT the head can potentially be
brought into INT8 without accuracy regression.  Compare mAP with and without.

### Step 4 — QAT fine-tuning loop

```python
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = optim.SGD(pt_model.parameters(), lr=5e-5, momentum=0.937, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=30)

# Sparsity mask (built from the sparse Phase-A weights)
masks = build_sparsity_masks(pt_model, exclude_prefixes=("model.0", "model.23"))

pt_model.train()
for epoch in range(30):
    for imgs, targets in train_loader:
        imgs    = imgs.cuda()
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = pt_model(imgs, targets)   # YOLOv10 training forward
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()

        # Re-apply sparsity mask after weight update
        apply_masks(pt_model, masks)

    scheduler.step()
    mto.save(pt_model, f"checkpoints/sparse_qat_epoch{epoch}.pth")
```

**Note on YOLOv10 training forward**: Ultralytics' `model.train()` manages the data
pipeline internally.  For the custom loop, use `ultralytics.data.build_dataloader` or
implement a COCO-compatible `torch.utils.data.DataLoader`.  The loss function is
accessible via `model.model.model` → set model to training mode and pass `(imgs, batch)`
matching Ultralytics' internal format.  An alternative is to patch Ultralytics'
`Trainer.optimizer_step()` to call `apply_masks()` — this lets the Ultralytics training
loop run unchanged while maintaining the sparsity constraint.

### Step 5 — ONNX export with QDQ nodes

```python
# After QAT is complete — do NOT call mtq.fold_weight() before this
import torch

pt_model.eval()
dummy = torch.zeros(1, 3, 640, 640).cuda()

torch.onnx.export(
    pt_model,
    dummy,
    "exported_models/MDV6-yolov10-e_sparse_qat_640.onnx",
    opset_version=18,
    input_names=["images"],
    output_names=["output0"],
    do_constant_folding=True,
)
```

The resulting ONNX will have `QuantizeLinear` / `DequantizeLinear` nodes encoding the
QAT-tuned scales — identical format to our current PTQ ONNX.  Our existing
`trt_export.py` consumes it unchanged.

---

## 6. Export and engine rebuild

Once the QAT ONNX exists, the TRT engine build is identical to the current sparse pipeline:

```bash
./run_export_sparse_qat.sh
# (new script, identical to run_export_e640_int8_sparse.sh but pointing at the QAT ONNX
#  instead of deriving QDQ from the dense .pt via our existing quantizer)
```

Or pass the QAT ONNX directly as `--onnx_override` to the export tool (new argument to
add when implementing this).

The `--sparse_weights` flag remains: it sets `BuilderFlag::SPARSE_WEIGHTS` at engine-build
time so TRT selects sparse Tensor Core kernels on Ampere.

---

## 7. Evaluation checkpoints

Run mAP evaluation at each phase gate.  Use the WCS val split + COCO val (converted).

| Phase gate | Expected mAP@50 vs. dense FP16 baseline |
|------------|------------------------------------------|
| Dense INT8 PTQ (current)          | -1 to -3 pp  |
| After 2:4 pruning (no fine-tune)  | -5 to -15 pp (magnitude-dependent) |
| Phase A: sparse fine-tune done    | within -1 to -2 pp of dense baseline |
| Phase B: sparse QAT done          | within -1 pp of dense INT8 PTQ |
| **Go/no-go**: proceed to production | < -2 pp vs. dense INT8, > 15% TRT speedup on Ampere |

---

## 8. Optimization deep dive

This section covers additional optimization pathways via NVIDIA ModelOpt that become
practical once we have a labeled fine-tuning dataset and an established training loop.
All of these are *post-sparse-QAT* options — they require the dataset and infrastructure
from §2–5 before becoming feasible.

---

### 8.1 FP8 quantization (W8A8 in FP8)

**Hardware requirement**: Ada Lovelace (RTX 4090 / L40S) or Hopper (H100).
**TRT requirement**: TRT 10+, opset 21 ONNX.

```python
import modelopt.torch.quantization as mtq
model = mtq.quantize(model, mtq.FP8_DEFAULT_CFG, calibrate_fn)
torch.onnx.export(model, ..., opset_version=21)
```

**Why it's interesting**: FP8 (E4M3) gives the same 2× math throughput as INT8 on Ada/
Hopper Tensor Cores, but with much higher dynamic range — no per-tensor scale sensitivity,
no head-exclusion requirement.  If the hardware target is Ada or later, FP8 may achieve
the same speed as INT8 with closer to FP16 accuracy and without the quantization
sensitivity analysis that INT8 requires.

**Exclusion control**: Same `quant_cfg` wildcard mechanism as INT8.  Given FP8's better
dynamic range, it is worth *not* excluding model.23 and measuring accuracy directly.

**INT8 + FP8 comparison**: On Ampere (RTX 3090, A100) FP8 has no hardware support.
On Ada it should match or beat INT8 accuracy while being equivalent in throughput.
FP8 + sparse is not a documented validated combination but is theoretically layerable.

---

### 8.2 INT4 weight-only quantization (W4A16)

**Hardware requirement**: Ampere or later (works on RTX 3090/A100/A10G).
**TRT requirement**: TRT 10+, opset 21 ONNX.

```python
import modelopt.torch.quantization as mtq
model = mtq.quantize(model, mtq.INT4_AWQ_CFG, calibrate_fn)
```

**What it does**: Conv weights are stored and computed as INT4; activations remain FP16.
This is "weight-only" quantization — primarily a memory-bandwidth optimization, not a
math-throughput optimization.  For YOLO inference (activation-bound), the benefit is
smaller than INT8 full quantization.

**When to use**: If the bottleneck is weight loading (large models, batch-1 inference, CPU
prefetch bound), INT4 can reduce model size by 4× vs. FP16 and may improve throughput
on memory-bandwidth-limited deployments.  For Frigate-style batch-1 real-time inference,
this is plausible.

**Exclusion**: Same wildcard mechanism.  The detection head (model.23) should still be
excluded unless QAT recovery demonstrates it's safe.

**AWQ algorithm** (`awq_lite`): Activation-Weighted Quantization — scales input channels
before quantizing weights to minimize per-channel quantization error.  Lower accuracy
loss than naive INT4 without calibration.

---

### 8.3 Structured channel pruning via FastNAS

**Hardware requirement**: Any (runs on same GPU used for training).
**Expected speedup**: ~1.5–2× throughput for 50% FLOPs reduction.

FastNAS searches for optimal per-layer channel widths (output channel count) across the
whole network in a single pass.  It respects skip connections — paired Conv2d layers
across residual paths are constrained to have matching widths automatically.

```python
import modelopt.torch.prune as mtp

# Exclude detection head from pruning (class priors baked into head weights)
ss_config = mtp.fastnas.FastNASConfig()
ss_config["nn.Conv2d"]["*model.23*"] = None   # freeze head at full width
ss_config["nn.Conv2d"]["*model.0*"]  = None   # keep stem (3-channel, limited flexibility)
ss_config["nn.Conv2d"]["*"]["channel_divisor"] = 8   # align to 8 for Tensor Core

def score_fn(model):
    return evaluate_map50(model, val_loader)  # returns float

model, _ = mtp.prune(
    model,
    mode="fastnas",
    constraints={"flops": "50%"},
    dummy_input=(torch.zeros(1, 3, 640, 640).cuda(),),
    config={"score_func": score_fn, "data_loader": val_loader},
)
```

**Workflow after pruning**:
1. `mtp.prune()` returns a *sliced* model with the narrow channel widths applied.
2. Fine-tune for 50–100 epochs (use `mtd.convert()` KD loss with the original dense model
   as teacher — strongly recommended for mAP recovery; see §8.4).
3. Optionally apply INT8/FP8 quantization on the pruned model.
4. Export to ONNX → TRT as normal.

**Interaction with 2:4 sparsity**: A pruned model (fewer channels) combined with 2:4
sparsity (within remaining channels) is a reasonable combination on Ampere+.  The pruning
reduces total FLOPs first; sparsity then halves the math for remaining Conv layers.
Combined: ~4× reduction vs. dense FP16 baseline.

**YOLOv10-specific considerations**:
- C2f blocks use bottleneck ratios.  FastNAS should handle these as it did ResNet
  bottlenecks, but validation is required.
- The detection head (model.23) has class-specific weight priors — do not prune it.
- The SPPF pool (model.10) is a single pointwise + max-pool block; pruning it may
  disproportionately hurt mAP.  Consider excluding it as well.

---

### 8.4 Knowledge distillation during fine-tuning

**Hardware requirement**: Same as training.
**When to use**: Phase A sparse fine-tuning, post-FastNAS fine-tuning, and/or Phase B QAT.

```python
import modelopt.torch.distill as mtd

teacher_model = load_original_dense_fp32_model()  # frozen, eval mode
teacher_model.eval()

distillation_config = {
    "teacher_model": teacher_model,
    "criterion": mtd.LogitsDistillationLoss(),   # KL divergence on output logits
    "loss_balancer": mtd.StaticLossBalancer(),
}

student_model = mtd.convert(student_model, mode="kd_loss", config=distillation_config)

# Training loop — add KD loss to task loss
for imgs, targets in train_loader:
    task_loss = compute_detection_loss(student_model, imgs, targets)
    kd_loss   = student_model.compute_kd_loss()
    total_loss = task_loss + 0.5 * kd_loss     # tune the 0.5 weight
    total_loss.backward()
    optimizer.step()
```

**When KD matters most**: After structural pruning (FastNAS) where the output distribution
changes significantly.  For 2:4 sparse fine-tuning the weight change is smaller and KD
is a secondary benefit.

---

### 8.5 Combined optimization matrix

The table below shows practical priority ordering given our current hardware (Turing for
dev, Ampere for production).

| Optimization | Ampere benefit | Ada benefit | Complexity | Priority |
|--------------|---------------|-------------|------------|----------|
| 2:4 sparse + PTQ INT8 (current) | ✓ ~1.5–2× | ✓ ✓ 2–3× | low | **now** |
| 2:4 sparse + QAT INT8 (Phase B) | ✓✓ same speed + better mAP | ✓✓ | medium | **after Ampere test** |
| FastNAS 50% FLOP prune + INT8   | ✓✓ 2–3× total | ✓✓✓ | high | after data confirmed |
| FP8 PTQ                         | ✗ no FP8 HW  | ✓✓ | low  | if Ada acquired |
| INT4 W4A16                      | ✓ bandwidth   | ✓  | medium | edge/memory-bound |
| Prune + sparse + QAT            | ✓✓✓ 3–5×     | ✓✓✓✓ | very high | post-validation |

---

### 8.6 Head quantization: can QAT "fix" it?

Today we exclude model.23 from INT8 because PTQ calibration scales for the detection head
cause mAP regression.  The root cause is that the head outputs are highly sensitive to
scale choice — the objectness and class confidence outputs are used for NMS thresholding,
and a coarse scale shifts the effective threshold.

With QAT fine-tuning, the model can *learn* to compensate for the quantization error in
the head by adjusting both the scales and the preceding weights.  This is the standard
QAT value proposition.  The recommendation is:

1. Start Phase B with head excluded (safe baseline, matching current PTQ quality).
2. After 10 epochs, enable head quantizers: `mtq.enable_quantizer(model, "*model.23*")`.
3. Continue for 20 more epochs.
4. Evaluate: if mAP is within -1 pp of the excluded-head baseline, proceed with head
   included (extra Tensor Core coverage on the head layers = marginal throughput gain).

If the head cannot be brought into INT8 even with QAT, keep it excluded — the throughput
cost is small (~5% of total layer time).

---

*This plan is contingent on Ampere sparsity test results showing >15% throughput
improvement over dense INT8 baseline.  If the Ampere gain is marginal (<10%), reassess
whether the fine-tuning investment is justified and consider FP8 (requires Ada hardware)
or FastNAS channel pruning as alternative first steps.*
