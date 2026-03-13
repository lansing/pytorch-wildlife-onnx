"""
prune_train.py
--------------
Structured channel pruning + fine-tuning for MDV6-yolov10.

Uses torch-pruning (https://github.com/VainF/Torch-Pruning) for dependency-
aware magnitude-based channel pruning.  torch-pruning correctly traces
YOLOv10's C2f skip connections and shared-channel constraints across concat
operations, which ModelOpt FastNAS's graph analyzer does not support.

Workflow
--------
1. Load the dense MDV6-yolov10-c (or -e) via YoloV9Loader.
2. Build a channel-importance ranking (L2-norm of output channels, global).
3. Prune: remove the least-important channels until the FLOPs target is met,
   keeping model.0 / model.10 / model.23 at full width.
4. Fine-tune the sliced model on megadetector_ft training split to recover mAP.
5. Export the pruned model to a float32 ONNX file (best val-loss checkpoint).
6. Save a plain PyTorch checkpoint for resuming or Phase 2 QAT.

After this script completes, compile to TRT:
    make export-pruned
    # or directly:
    python -m PytorchWildlife_Export.export_tool \\
        --model_type yolov10 --model_version MDV6-yolov10-c \\
        --onnx-override /app/checkpoints/prune_c640/MDV6-yolov10-c_pruned.onnx \\
        --format float16 --runtime tensorrt \\
        --uint8_input --nhwc_input --denormalized_input \\
        --output_path /exported_models/MDV6-yolov10-c_pruned_float16_640_denorm_nhwc_uint8input.engine

Exclusions (kept at full width)
--------------------------------
    model.0   — 3-channel RGB input conv; no Tensor-Core path for 3-ch inputs
    model.10  — SPPF / PSA attention block; pruning may hurt recall disproportionately
    model.23  — detection head; class priors are baked into head weights

Channel divisor: 8 (Tensor Core alignment on Turing/Ampere)

Requires: pip install torch-pruning
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from .dataloader import build_md_dataloader, to_device

LOGGER = logging.getLogger(__name__)

# Indices of DetectionModel.model layers excluded from pruning
# Only the RGB-input stem (model.0) is excluded from pruning.
# model.23 (detection head) is NOT excluded: torch-pruning automatically protects
# output tensors, so the final Conv outputs of the head's cv2/cv3 branches are safe.
# Including model.23 in ignored_layers would cascade the "no pruning" constraint
# backward through the entire FPN neck, resulting in 0% FLOPs reduction.
_EXCLUDED_LAYER_INDICES = [0]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _cfg_get(cfg: dict, key: str, default=None):
    keys = key.split(".")
    v = cfg
    for k in keys:
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_ultralytics_model(model_version: str, weights_path: Optional[str] = None):
    """Load MDV6-yolov10 via YoloV9Loader and patch args/hyp for fine-tuning."""
    import types
    from PytorchWildlife_Export.model_loaders.yolov9_loader import YoloV9Loader

    loader = YoloV9Loader(version=model_version, device="cuda", weights=weights_path)
    ul_model = loader.load_model()
    detection_model = ul_model.model.cuda().float()

    _TRAIN_DEFAULTS = dict(box=7.5, cls=0.5, dfl=1.5)
    for attr in ("args", "hyp"):
        obj = getattr(detection_model, attr, None)
        if isinstance(obj, dict):
            merged = {**_TRAIN_DEFAULTS, **obj}
            setattr(detection_model, attr, types.SimpleNamespace(**merged))

    return ul_model, detection_model


# ---------------------------------------------------------------------------
# Manual C2f-internal channel pruning
# ---------------------------------------------------------------------------
# torch-pruning's FX tracer cannot handle YOLOv10's dynamic routing (y[m.f]
# list indexing in _forward_once), returning 0 prunable groups.  We instead
# prune the INTERNAL hidden channels of each C2f block directly.
#
# C2f forward (simplified):
#   y = list(cv1(x).chunk(2, 1))      # [chunk0(c), chunk1(c)]
#   y += [m(y[-1]) for m in self.m]   # bottleneck chain, each c→c
#   return cv2(cat(y, 1))             # (2+n)*c → c2
#
# Pruning c → new_c is SELF-CONTAINED: the block's external interface (c1, c2)
# doesn't change, so no cross-layer dependency tracking is needed.
# ---------------------------------------------------------------------------

def _prune_conv_module(conv_mod, in_keep, out_keep):
    """Prune in/out channels of an Ultralytics Conv (conv + optional BN).

    conv_mod : Ultralytics Conv object with .conv (nn.Conv2d) and optionally .bn
    in_keep  : 1-D LongTensor of input channel indices to keep, or None
    out_keep : 1-D LongTensor of output channel indices to keep, or None
    """
    old_c = conv_mod.conv
    has_bias = old_c.bias is not None
    new_in  = len(in_keep)  if in_keep  is not None else old_c.in_channels
    new_out = len(out_keep) if out_keep is not None else old_c.out_channels

    # Determine group count — depthwise convs must preserve groups == channels
    groups = old_c.groups
    if groups > 1:
        # depthwise: groups == in_channels; we never prune those here
        groups = new_in

    dev = old_c.weight.device
    new_conv = nn.Conv2d(
        new_in, new_out,
        old_c.kernel_size, old_c.stride, old_c.padding,
        dilation=old_c.dilation, groups=groups, bias=has_bias,
    ).to(dev)
    with torch.no_grad():
        w = old_c.weight.data
        if out_keep is not None:
            w = w[out_keep]
        if in_keep is not None and old_c.groups == 1:
            w = w[:, in_keep]
        new_conv.weight.copy_(w)
        if has_bias:
            b = old_c.bias.data
            if out_keep is not None:
                b = b[out_keep]
            new_conv.bias.copy_(b)
    conv_mod.conv = new_conv

    if hasattr(conv_mod, "bn") and conv_mod.bn is not None:
        old_bn = conv_mod.bn
        new_bn = nn.BatchNorm2d(
            new_out, eps=old_bn.eps, momentum=old_bn.momentum,
            affine=old_bn.affine, track_running_stats=old_bn.track_running_stats,
        ).to(dev)
        if old_bn.affine:
            with torch.no_grad():
                idx = out_keep if out_keep is not None else torch.arange(new_out, device=dev)
                new_bn.weight.copy_(old_bn.weight[idx])
                new_bn.bias.copy_(old_bn.bias[idx])
        if old_bn.track_running_stats:
            with torch.no_grad():
                idx = out_keep if out_keep is not None else torch.arange(new_out, device=dev)
                new_bn.running_mean.copy_(old_bn.running_mean[idx])
                new_bn.running_var.copy_(old_bn.running_var[idx])
                new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)
        conv_mod.bn = new_bn


def _is_ultralytics_conv(mod) -> bool:
    """Return True if mod is an Ultralytics Conv wrapper with .conv and .bn attrs."""
    return hasattr(mod, "conv") and isinstance(mod.conv, nn.Conv2d)


def _prune_c2f_hidden(c2f, pruning_ratio: float, round_to: int = 8):
    """Prune the hidden channel width of a C2f block in-place.

    Reduces c2f.c from c → new_c without changing the block's external
    input channel (c1) or output channel (c2) interface.

    Skips blocks whose bottleneck modules are not standard Ultralytics Conv
    wrappers (e.g. C2fCIB's CIB blocks which use nn.Sequential internally).
    """
    # Verify cv1 and cv2 are standard Ultralytics Conv objects
    if not _is_ultralytics_conv(c2f.cv1) or not _is_ultralytics_conv(c2f.cv2):
        return  # non-standard C2f variant — skip

    # Verify all bottleneck modules have standard .cv1/.cv2 Ultralytics Conv attrs
    for bn_mod in c2f.m:
        if not (hasattr(bn_mod, "cv1") and _is_ultralytics_conv(bn_mod.cv1) and
                hasattr(bn_mod, "cv2") and _is_ultralytics_conv(bn_mod.cv2)):
            return  # CIB-style or other non-standard bottleneck — skip whole block

    c = c2f.c
    new_c = max(round_to, int(c * (1.0 - pruning_ratio) / round_to) * round_to)
    if new_c >= c:
        return  # nothing to prune

    n_bn = len(c2f.m)

    # ---- Importance: L2 norm of cv1 output channels ----
    cv1_w = c2f.cv1.conv.weight.data  # (2*c, c1, kH, kW)
    imp = cv1_w.norm(p=2, dim=(1, 2, 3))

    # First c channels → chunk0 (pass-through to cv2 directly)
    first_keep = imp[:c].topk(new_c).indices.sort().values          # (new_c,)
    # Second c channels → chunk1 (fed into bottleneck chain)
    second_keep = imp[c:].topk(new_c).indices.sort().values         # (new_c,)
    cv1_out_keep = torch.cat([first_keep, second_keep + c])         # (2*new_c,)

    # ---- Prune cv1: 2*c → 2*new_c ----
    _prune_conv_module(c2f.cv1, in_keep=None, out_keep=cv1_out_keep)

    # ---- Prune each Bottleneck ----
    # C2f data flow (bottleneck chain):
    #   chunk1 (new_c channels from cv1's second half) → bn[0] → bn[1] → ...
    # bn[0].cv1 must accept new_c channels (specifically the second_keep subset of original c).
    # bn[i].cv1 for i>0 must accept new_c channels (the bn_cv2_out_keeps[i-1] subset of orig c).
    bn_cv2_out_keeps = []
    prev_in_keep = second_keep   # channels bn[0].cv1 receives (subset of original c)

    for bn_mod in c2f.m:
        # bn.cv1: original (old_bn_c, c, kH, kW) → new (new_bn_c, new_c, kH, kW)
        bn_cv1_w = bn_mod.cv1.conv.weight.data          # (old_bn_c, old_c, kH, kW)
        old_bn_c = bn_cv1_w.shape[0]
        new_bn_c = max(round_to, int(old_bn_c * (1.0 - pruning_ratio) / round_to) * round_to)
        bn_cv1_out_keep = bn_cv1_w.norm(p=2, dim=(1, 2, 3)).topk(new_bn_c).indices.sort().values

        # in_keep=prev_in_keep: prune the input from c → new_c, keeping prev_in_keep channels
        _prune_conv_module(bn_mod.cv1, in_keep=prev_in_keep, out_keep=bn_cv1_out_keep)

        # bn.cv2: original (old_c, old_bn_c, kH, kW) → new (new_c, new_bn_c, kH, kW)
        bn_cv2_w = bn_mod.cv2.conv.weight.data          # (old_c, old_bn_c, kH, kW)
        bn_cv2_out_keep = bn_cv2_w.norm(p=2, dim=(1, 2, 3)).topk(new_c).indices.sort().values

        _prune_conv_module(bn_mod.cv2, in_keep=bn_cv1_out_keep, out_keep=bn_cv2_out_keep)
        bn_cv2_out_keeps.append(bn_cv2_out_keep)

        # Next bottleneck receives this bottleneck's output channels (indexed in original c space)
        prev_in_keep = bn_cv2_out_keep

    # ---- Build cv2 input indices ----
    # Original C2f cv2 takes cat of: chunk0(c) | chunk1(c) | bn0_out(c) | ... | bnn_out(c)
    # Each segment is c-wide in the original tensor.  We kept new_c channels per segment:
    #   chunk0: first_keep (indices within 0..c-1)
    #   chunk1: second_keep + c (indices within c..2c-1)
    #   bn[i]:  bn_cv2_out_keeps[i] + (2+i)*c (indices within (2+i)*c..(3+i)*c-1)
    cv2_in_keep_parts = [
        first_keep,                                   # chunk0 kept channels (offset 0)
        second_keep + c,                              # chunk1 kept channels (offset c)
    ]
    for i, bk in enumerate(bn_cv2_out_keeps):
        cv2_in_keep_parts.append(bk + (2 + i) * c)  # bn[i] kept channels (offset (2+i)*c)

    cv2_in_keep = torch.cat(cv2_in_keep_parts)       # (2+n)*new_c indices

    # ---- Prune cv2 input: (2+n)*c → (2+n)*new_c, output unchanged ----
    _prune_conv_module(c2f.cv2, in_keep=cv2_in_keep, out_keep=None)

    # Update hidden channel count
    c2f.c = new_c


def prune_model(
    detection_model: nn.Module,
    device: str,
    flops_fraction: float = 0.50,
    channel_divisor: int = 8,
    excluded_layer_indices: list[int] = _EXCLUDED_LAYER_INDICES,
) -> nn.Module:
    """Apply one-shot magnitude-based structured channel pruning to all C2f blocks.

    Prunes the INTERNAL hidden channels of each C2f block without changing its
    external input/output interface.  This approach bypasses torch-pruning's FX
    tracer (which fails on YOLOv10's dynamic routing) by working directly on the
    known C2f module structure.

    Parameters
    ----------
    detection_model       : DetectionModel on CUDA
    device                : "cuda" or "cpu"
    flops_fraction        : target FLOPs fraction *remaining* (0.50 = keep 50%)
    channel_divisor       : output channel alignment (8 for Tensor Core)
    excluded_layer_indices: DetectionModel.model layer indices to skip entirely
    """
    detection_model.eval()
    example_inputs = torch.zeros(1, 3, 640, 640, device=device)

    # FLOPs measurement helper — always restores model device afterwards because
    # tp.utils.count_ops_and_params may move the model to CPU internally.
    def _count_macs(m):
        saved_device = next(m.parameters()).device
        try:
            import torch_pruning as tp
            macs, params = tp.utils.count_ops_and_params(m, example_inputs)
            return macs, params
        except Exception:
            return None, None
        finally:
            m.to(saved_device)

    base_macs, base_params = _count_macs(detection_model)
    LOGGER.info(
        "Baseline: FLOPs=%.2f G  Params=%.2f M",
        (base_macs or 0) / 1e9, (base_params or 0) / 1e6,
    )

    # Approximate channel pruning ratio to hit the FLOPs target.
    # C2f bottleneck FLOPs ∝ c², so (1-p)² = flops_fraction → p = 1 - sqrt(flops_fraction)
    pruning_ratio = max(0.05, min(0.70, 1.0 - math.sqrt(flops_fraction)))
    LOGGER.info(
        "C2f hidden-channel pruning ratio: %.2f  (targeting %.0f%% FLOPs remaining)",
        pruning_ratio, flops_fraction * 100,
    )

    excluded_set = set(excluded_layer_indices)
    c2f_type_names = {"C2f", "C2fCIB", "C2fAttn"}
    n_pruned = 0

    for layer_idx, layer in enumerate(detection_model.model):
        if layer_idx in excluded_set:
            continue
        # Prune every C2f-family block found in this top-level layer
        for submod_name, submod in layer.named_modules():
            if type(submod).__name__ in c2f_type_names:
                old_c = submod.c
                _prune_c2f_hidden(submod, pruning_ratio, round_to=channel_divisor)
                new_c = submod.c
                if new_c < old_c:
                    LOGGER.info(
                        "  layer[%d].%s  %s: c %d → %d",
                        layer_idx, submod_name, type(submod).__name__, old_c, new_c,
                    )
                    n_pruned += 1

    LOGGER.info("Pruned %d C2f blocks.", n_pruned)

    # Measure FLOPs on pruned model (_count_macs restores device internally).
    pruned_macs, pruned_params = _count_macs(detection_model)
    if base_macs and pruned_macs:
        LOGGER.info(
            "Pruned:   FLOPs=%.2f G (%.0f%% of baseline)  Params=%.2f M (%.0f%% of baseline)",
            pruned_macs / 1e9, pruned_macs / base_macs * 100,
            pruned_params / 1e6, pruned_params / base_params * 100,
        )

    # Final device restoration: ensure model and all unregistered tensor attrs are on
    # the target device.  _count_macs.finally() handles registered params/buffers, but
    # unregistered __dict__ tensor attrs (self.anchors, self.strides set during eval
    # forward passes) are not moved by .to().  Walk and fix them now.
    detection_model.to(device)
    target_device = torch.device(device)
    _param_ids = {id(p) for p in detection_model.parameters()}
    _buf_ids   = {id(b) for b in detection_model.buffers()}
    _skip_ids  = _param_ids | _buf_ids
    for _mod in detection_model.modules():
        for _attr, _val in list(vars(_mod).items()):
            if isinstance(_val, torch.Tensor) and id(_val) not in _skip_ids:
                if _val.device != target_device:
                    setattr(_mod, _attr, _val.to(target_device))

    # Drop any pre-initialized criterion: _count_macs runs model.loss() internally
    # which lazily creates detection_model.criterion while the model may be on CPU,
    # leaving criterion.proj etc. on CPU.  Delete it so it gets re-initialized fresh
    # (on the correct device) during the first real training step.
    if hasattr(detection_model, "criterion"):
        del detection_model.criterion

    detection_model.train()
    return detection_model


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def _cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    total_epochs: int,
    lr_init: float,
    lr_min: float,
) -> float:
    if total_epochs <= 1:
        lr = lr_init
    else:
        lr = lr_min + 0.5 * (lr_init - lr_min) * (
            1 + math.cos(math.pi * epoch / total_epochs)
        )
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 10.0,
    log_interval: int = 50,
) -> dict:
    model.train()
    total_loss = 0.0
    totals = {"box": 0.0, "cls": 0.0, "dfl": 0.0}
    n_batches = 0

    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        batch["img"] = batch["img"].float() / 255.0

        loss, loss_items = model.loss(batch)
        if loss.ndim > 0:
            loss = loss.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        if loss_items is not None and len(loss_items) >= 3:
            totals["box"] += loss_items[0].item()
            totals["cls"] += loss_items[1].item()
            totals["dfl"] += loss_items[2].item()
        n_batches += 1

        if (i + 1) % log_interval == 0:
            LOGGER.info(
                "  [%4d/%4d]  loss=%.4f  box=%.4f  cls=%.4f  dfl=%.4f",
                i + 1, len(loader),
                total_loss / n_batches,
                totals["box"] / n_batches,
                totals["cls"] / n_batches,
                totals["dfl"] / n_batches,
            )

    n = max(n_batches, 1)
    return {
        "loss":     total_loss / n,
        "box_loss": totals["box"] / n,
        "cls_loss": totals["cls"] / n,
        "dfl_loss": totals["dfl"] / n,
    }


def val_loss_epoch(
    model: nn.Module,
    loader,
    device: str,
    max_batches: int = 0,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            batch = to_device(batch, device)
            batch["img"] = batch["img"].float() / 255.0
            loss, _ = model.loss(batch)
            if loss.ndim > 0:
                loss = loss.sum()
            total += loss.item()
            n += 1
    model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# ONNX export of the pruned model
# ---------------------------------------------------------------------------

def export_pruned_onnx(
    detection_model: nn.Module,
    onnx_path: str,
    input_shape: tuple = (1, 3, 640, 640),
    opset: int = 18,
    simplify: bool = True,
    device: str = "cuda",
) -> str:
    """Export the pruned DetectionModel to a float32 ONNX file.

    Sets YOLOv10 detection head layers to export mode so the forward returns
    the NMS-ready (1, 300, 6) tensor expected by our eval and export pipelines.
    """
    import onnx as _onnx

    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    detection_model.eval()

    # Set YOLOv10 head to export mode → returns (1, 300, 6) output
    for m in detection_model.modules():
        if hasattr(m, "export"):
            m.export = True
        if hasattr(m, "format"):
            m.format = "onnx"

    dummy = torch.zeros(*input_shape, device=device)
    LOGGER.info("Exporting pruned ONNX: %s  input=%s", onnx_path, input_shape)

    try:
        torch.onnx.export(
            detection_model,
            dummy,
            onnx_path,
            opset_version=opset,
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
        )
    finally:
        for m in detection_model.modules():
            if hasattr(m, "export"):
                m.export = False

    model_proto = _onnx.load(onnx_path)
    _onnx.checker.check_model(model_proto)
    LOGGER.info("ONNX graph check passed.")

    if simplify:
        try:
            import onnxslim
            model_proto = onnxslim.slim(onnx_path)
            _onnx.save(model_proto, onnx_path)
            LOGGER.info("ONNX simplified with onnxslim.")
        except Exception as exc:
            LOGGER.warning("onnxslim failed (%s) — using unsimplified graph.", exc)

    LOGGER.info("Pruned ONNX saved: %s", onnx_path)
    return onnx_path


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_prune_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    """Save pruned model checkpoint (full model + optimizer state)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch":     epoch,
            "metrics":   metrics,
            "model":     model,          # full module (pruned architecture + weights)
            "optimizer": optimizer.state_dict(),
        },
        str(path),
    )
    LOGGER.info("Pruned checkpoint saved: %s", path)


def load_prune_checkpoint(path: Path) -> tuple[nn.Module, dict]:
    """Load a pruned-model checkpoint.  Returns (model, metadata)."""
    ckpt = torch.load(str(path), map_location="cuda", weights_only=False)
    model = ckpt["model"].cuda().float()
    meta = {k: ckpt[k] for k in ("epoch", "metrics") if k in ckpt}
    LOGGER.info(
        "Pruned checkpoint loaded: %s  (epoch %s)", path, meta.get("epoch", "?")
    )
    return model, meta


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Structured channel pruning + fine-tuning for MDV6-yolov10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",     required=True, help="Path to config_prune_c640.yaml")
    parser.add_argument("--epochs",     type=int, default=None, help="Override fine-tuning epochs")
    parser.add_argument("--skip-prune", action="store_true",
                        help="Skip pruning search and load latest checkpoint for fine-tuning resume")
    parser.add_argument("--log-level",  default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config(args.config)

    # -----------------------------------------------------------------------
    # Config
    # -----------------------------------------------------------------------
    model_version   = _cfg_get(cfg, "model.version",      "MDV6-yolov10-c")
    weights_path    = _cfg_get(cfg, "model.weights",      None)
    dataset_yaml    = _cfg_get(cfg, "data.dataset_yaml")
    imgsz           = _cfg_get(cfg, "data.imgsz",         640)
    batch_size      = _cfg_get(cfg, "data.batch_size",    8)
    workers         = _cfg_get(cfg, "data.workers",       4)
    flops_fraction  = _cfg_get(cfg, "prune.flops_fraction",  0.50)
    channel_divisor = _cfg_get(cfg, "prune.channel_divisor", 8)
    epochs          = args.epochs if args.epochs is not None else _cfg_get(cfg, "train.epochs", 60)
    lr_init         = _cfg_get(cfg, "train.lr",           5e-5)
    lr_min          = _cfg_get(cfg, "train.lr_min",       1e-6)
    weight_decay    = _cfg_get(cfg, "train.weight_decay", 1e-4)
    grad_clip       = _cfg_get(cfg, "train.grad_clip",    10.0)
    log_interval    = _cfg_get(cfg, "train.log_interval", 50)
    val_batches     = _cfg_get(cfg, "train.val_batches",  50)
    ckpt_dir        = Path(_cfg_get(cfg, "output.checkpoint_dir", "checkpoints/prune"))
    onnx_out        = _cfg_get(cfg, "output.onnx_path",
                               str(ckpt_dir / f"{model_version}_pruned.onnx"))

    if dataset_yaml is None:
        parser.error("data.dataset_yaml must be set in config")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Device: %s", device)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    LOGGER.info("Building data loaders from %s", dataset_yaml)
    train_loader = build_md_dataloader(
        dataset_yaml, split="train", imgsz=imgsz,
        batch_size=batch_size, workers=workers, augment=True,
    )
    val_loader = build_md_dataloader(
        dataset_yaml, split="val", imgsz=imgsz,
        batch_size=batch_size, workers=0, augment=False,
    )  # workers=0: avoid multiprocessing deadlock when breaking mid-iteration
    LOGGER.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    # -----------------------------------------------------------------------
    # Pruning (or resume from checkpoint)
    # -----------------------------------------------------------------------
    start_epoch = 0
    optimizer_state = None

    if args.skip_prune:
        ckpt_glob = sorted(ckpt_dir.glob("prune_*.pt"))
        if not ckpt_glob:
            parser.error(f"--skip-prune: no .pt checkpoint found in {ckpt_dir}")
        latest = ckpt_glob[-1]
        LOGGER.info("--skip-prune: restoring from %s", latest)
        detection_model, meta = load_prune_checkpoint(latest)
        start_epoch = meta.get("epoch", 0) + 1
        optimizer_state = None  # optimizer state saved separately if needed
        LOGGER.info("Resuming fine-tuning from epoch %d", start_epoch)

        # Patch args/hyp for the restored model (may have been cleared)
        import types
        _TRAIN_DEFAULTS = dict(box=7.5, cls=0.5, dfl=1.5)
        for attr in ("args", "hyp"):
            obj = getattr(detection_model, attr, None)
            if isinstance(obj, dict):
                merged = {**_TRAIN_DEFAULTS, **obj}
                setattr(detection_model, attr, types.SimpleNamespace(**merged))
    else:
        # Fresh start: load dense model and prune
        _, detection_model = load_ultralytics_model(model_version, weights_path)

        LOGGER.info(
            "Starting channel pruning (target FLOPs fraction=%.0f%%)...",
            flops_fraction * 100,
        )
        detection_model = prune_model(
            detection_model,
            device=device,
            flops_fraction=flops_fraction,
            channel_divisor=channel_divisor,
        )

    # Re-enable gradients for all float parameters
    for p in detection_model.parameters():
        if p.is_floating_point():
            p.requires_grad_(True)
    n_trainable = sum(1 for p in detection_model.parameters() if p.requires_grad)
    LOGGER.info("Trainable parameters: %d", n_trainable)

    # -----------------------------------------------------------------------
    # Fine-tuning loop
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        detection_model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
    )
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as e:
            LOGGER.warning("Could not restore optimizer state: %s", e)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_onnx_exported = False

    LOGGER.info(
        "Fine-tuning: %d epochs  lr=%.2e → %.2e",
        epochs, lr_init, lr_min,
    )

    for epoch in range(start_epoch, start_epoch + epochs):
        lr = _cosine_lr(optimizer, epoch - start_epoch, epochs, lr_init, lr_min)
        LOGGER.info("Epoch %d/%d  lr=%.2e", epoch + 1, start_epoch + epochs, lr)

        metrics = train_one_epoch(
            detection_model, train_loader, optimizer,
            device=device, grad_clip=grad_clip, log_interval=log_interval,
        )
        LOGGER.info(
            "  Train  loss=%.4f  box=%.4f  cls=%.4f  dfl=%.4f",
            metrics["loss"], metrics["box_loss"], metrics["cls_loss"], metrics["dfl_loss"],
        )

        v_loss = val_loss_epoch(detection_model, val_loader, device, max_batches=val_batches)
        LOGGER.info("  Val    loss=%.4f", v_loss)
        metrics["val_loss"] = v_loss

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            ckpt_path = ckpt_dir / f"prune_{model_version}_best.pt"
            save_prune_checkpoint(detection_model, optimizer, epoch, metrics, ckpt_path)
            export_pruned_onnx(
                detection_model, onnx_out,
                input_shape=(1, 3, imgsz, imgsz), device=device,
            )
            best_onnx_exported = True
            LOGGER.info(
                "  New best val loss %.4f → checkpoint + ONNX updated: %s", best_val_loss, onnx_out
            )

    if not best_onnx_exported:
        LOGGER.warning("No improvement seen; exporting final epoch as ONNX.")
        export_pruned_onnx(
            detection_model, onnx_out,
            input_shape=(1, 3, imgsz, imgsz), device=device,
        )

    LOGGER.info("Done.  Best val loss: %.4f", best_val_loss)
    LOGGER.info("Pruned ONNX: %s", onnx_out)
    LOGGER.info(
        "\nNext step — compile to TRT float16:\n"
        "  make export-pruned\n"
        "  # or:\n"
        "  python -m PytorchWildlife_Export.export_tool \\\n"
        "    --model_type yolov10 --model_version %s \\\n"
        "    --onnx-override %s \\\n"
        "    --format float16 --runtime tensorrt \\\n"
        "    --uint8_input --nhwc_input --denormalized_input \\\n"
        "    --output_path /exported_models/%s_pruned_float16_%d_denorm_nhwc_uint8input.engine",
        model_version, onnx_out, model_version, imgsz,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
