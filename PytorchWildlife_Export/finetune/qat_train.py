"""
qat_train.py
------------
QAT fine-tuning loop for MDV6-yolov10.

Usage
-----
    # PTQ calibration only (baseline INT8, no gradient updates):
    python -m PytorchWildlife_Export.finetune.qat_train \
        --config config_qat.yaml --epochs 0

    # Standard QAT (head excluded from INT8):
    python -m PytorchWildlife_Export.finetune.qat_train \
        --config config_qat.yaml

    # Full INT8 QAT experiment (head included):
    python -m PytorchWildlife_Export.finetune.qat_train \
        --config config_qat.yaml --full-int8

    # Resume from checkpoint:
    python -m PytorchWildlife_Export.finetune.qat_train \
        --config config_qat.yaml --resume checkpoints/qat_epoch5.modelopt

Inside Docker:
    python3 -m PytorchWildlife_Export.finetune.qat_train --config /app/config_qat.yaml
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# ---------------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------------
from .dataloader import build_md_dataloader, to_device
from .quantize import (
    build_quant_config,
    calibrate_model,
    enable_head_quantizers,
    extract_clean_state_dict,
    extract_qat_scales,
    load_qat_checkpoint,
    save_qat_checkpoint,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def _cfg_get(cfg: dict, key: str, default=None):
    """Dot-notation access into nested config dict."""
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
    """Load an Ultralytics YOLO model and return the (ul_model, detection_model) pair.

    Parameters
    ----------
    model_version  : e.g. "MDV6-yolov10-e"
    weights_path   : explicit .pt path; if None, downloads via YoloV9Loader
    """
    import types
    from PytorchWildlife_Export.model_loaders.yolov9_loader import YoloV9Loader

    loader = YoloV9Loader(version=model_version, device="cuda", weights=weights_path)
    ul_model = loader.load_model()

    # ul_model.model is the raw DetectionModel nn.Module
    detection_model = ul_model.model
    detection_model = detection_model.cuda().float()

    # Ultralytics loss functions access training hyperparams via attribute syntax
    # (self.hyp.box, model.args.box etc).  When loaded from an inference checkpoint
    # these are plain dicts — convert to SimpleNamespace and fill any missing
    # training keys with YOLOv10 defaults so the loss criterion initialises cleanly.
    _TRAIN_DEFAULTS = dict(box=7.5, cls=0.5, dfl=1.5)
    for attr in ("args", "hyp"):
        obj = getattr(detection_model, attr, None)
        if isinstance(obj, dict):
            merged = {**_TRAIN_DEFAULTS, **obj}
            setattr(detection_model, attr, types.SimpleNamespace(**merged))

    return ul_model, detection_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _cosine_lr(optimizer, epoch: int, total_epochs: int, lr_init: float, lr_min: float):
    """Apply cosine LR decay in-place."""
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
    detection_model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    grad_clip: float = 10.0,
    log_interval: int = 50,
) -> dict:
    """Run one epoch of QAT fine-tuning.

    Returns
    -------
    dict with keys: loss, box_loss, cls_loss, dfl_loss  (all averages over epoch)
    """
    detection_model.train()
    total_loss = 0.0
    totals = {"box": 0.0, "cls": 0.0, "dfl": 0.0}
    n_batches = 0

    for i, batch in enumerate(loader):
        batch = to_device(batch, device)
        # Normalise images: Ultralytics DataLoader gives uint8 [0,255]
        batch["img"] = batch["img"].float() / 255.0

        loss, loss_items = detection_model.loss(batch)
        # Some Ultralytics versions return a 3-element vector (box/cls/dfl) instead of
        # a pre-summed scalar — reduce to scalar before backward.
        if loss.ndim > 0:
            loss = loss.sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(detection_model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        # loss_items: [box, cls, dfl] tensor (detached, already per-image)
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


def val_loss(
    detection_model: torch.nn.Module,
    loader,
    device: str,
    max_batches: int = 0,
) -> float:
    """Compute validation loss (no gradient).  Returns scalar."""
    detection_model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            batch = to_device(batch, device)
            batch["img"] = batch["img"].float() / 255.0
            loss, _ = detection_model.loss(batch)
            if loss.ndim > 0:
                loss = loss.sum()
            total += loss.item()
            n += 1
    detection_model.train()
    return total / max(n, 1)


# ---------------------------------------------------------------------------
# QAT output helpers
# ---------------------------------------------------------------------------

def _save_qat_outputs(
    detection_model: torch.nn.Module,
    ul_model,
    weights_path: str,
    scales_path: str,
    model_version: str = "",
) -> None:
    """Save QAT-finetuned float32 weights and learned INT8 scales.

    Produces two files:
      <weights_path>  — clean Ultralytics .pt (no ModelOpt quantizer state).
                        Feed to export_tool.py via --weights.
      <scales_path>   — JSON of QAT-learned activation scales per ONNX node.
                        Feed to export_tool.py via --scales-json.

    The export command to build the INT8 TRT engine from these outputs:
        python -m PytorchWildlife_Export.export_tool \\
            --model_type yolov10 \\
            --model_version MDV6-yolov10-e \\
            --weights    <weights_path> \\
            --scales_json <scales_path> \\
            --format int8 --runtime tensorrt \\
            --uint8_input --nhwc_input --denormalized_input \\
            --output_path /exported_models/MDV6-yolov10-e_int8_640_denorm_nhwc_uint8input.engine
    """
    import json

    weights_path = Path(weights_path)
    scales_path  = Path(scales_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Extract and save QAT-learned activation scales
    scales = extract_qat_scales(detection_model)
    with open(scales_path, "w") as f:
        json.dump(scales, f, indent=2)
    LOGGER.info("QAT scales saved: %s  (%d nodes)", scales_path, len(scales))

    # 2. Save clean float32 weights.
    # ul_model.model has QuantConv2d layers (not picklable), so we load a fresh
    # standard YOLO (from cache — no download), apply the clean state_dict, and save.
    clean_sd = extract_clean_state_dict(detection_model)
    from PytorchWildlife_Export.model_loaders.yolov9_loader import YoloV9Loader
    fresh_ul = YoloV9Loader(version=model_version, device="cpu").load_model()
    missing, unexpected = fresh_ul.model.load_state_dict(clean_sd, strict=False)
    if missing:
        LOGGER.warning("Missing keys when loading clean weights: %s", missing[:5])
    fresh_ul.save(str(weights_path))
    LOGGER.info("QAT finetuned weights saved: %s", weights_path)
    LOGGER.info(
        "\nNext step — export INT8 TRT engine with QAT scales:\n"
        "  python -m PytorchWildlife_Export.export_tool \\\n"
        "    --model_type yolov10 --model_version MDV6-yolov10-e \\\n"
        "    --weights %s \\\n"
        "    --scales_json %s \\\n"
        "    --format int8 --runtime tensorrt \\\n"
        "    --uint8_input --nhwc_input --denormalized_input \\\n"
        "    --output_path /exported_models/MDV6-yolov10-e_int8_qat_640_denorm_nhwc_uint8input.engine",
        weights_path, scales_path,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="QAT fine-tuning for MDV6-yolov10",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",  required=True, help="Path to config_qat.yaml")
    parser.add_argument("--resume",  default=None,  help="ModelOpt checkpoint to resume from")
    parser.add_argument("--full-int8", action="store_true",
                        help="Re-enable head quantizers after calibration (full INT8 experiment)")
    parser.add_argument("--epochs",  type=int, default=None,
                        help="Override number of training epochs from config")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config(args.config)

    # -----------------------------------------------------------------------
    # Hyper-parameters from config (with sensible defaults)
    # -----------------------------------------------------------------------
    model_version  = _cfg_get(cfg, "model.version",  "MDV6-yolov10-e")
    weights_path   = _cfg_get(cfg, "model.weights",  None)
    dataset_yaml   = _cfg_get(cfg, "data.dataset_yaml")
    imgsz          = _cfg_get(cfg, "data.imgsz",     640)
    batch_size     = _cfg_get(cfg, "data.batch_size", 8)
    workers        = _cfg_get(cfg, "data.workers",   4)
    calib_batches  = _cfg_get(cfg, "quant.calib_batches", 16)
    exclude_stem   = _cfg_get(cfg, "quant.exclude_stem",  True)
    exclude_head   = _cfg_get(cfg, "quant.exclude_head",  True)
    epochs         = args.epochs if args.epochs is not None else _cfg_get(cfg, "train.epochs", 10)
    lr_init        = _cfg_get(cfg, "train.lr",       1e-4)
    lr_min         = _cfg_get(cfg, "train.lr_min",   1e-6)
    weight_decay   = _cfg_get(cfg, "train.weight_decay", 1e-4)
    grad_clip      = _cfg_get(cfg, "train.grad_clip", 10.0)
    log_interval   = _cfg_get(cfg, "train.log_interval", 50)
    val_batches    = _cfg_get(cfg, "train.val_batches", 50)
    ckpt_dir       = Path(_cfg_get(cfg, "output.checkpoint_dir", "checkpoints"))
    weights_out    = _cfg_get(cfg, "output.weights_path", str(ckpt_dir / f"{model_version}_qat_finetuned.pt"))
    scales_out     = _cfg_get(cfg, "output.scales_json",  str(ckpt_dir / f"{model_version}_qat_scales.json"))

    if dataset_yaml is None:
        parser.error("data.dataset_yaml must be set in config")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("Device: %s", device)
    if device == "cpu":
        LOGGER.warning("CUDA not available — QAT will be very slow on CPU.")

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
    # Model
    # -----------------------------------------------------------------------
    ul_model, detection_model = load_ultralytics_model(model_version, weights_path)

    # -----------------------------------------------------------------------
    # PTQ calibration (or resume from checkpoint)
    # -----------------------------------------------------------------------
    start_epoch = 0
    optimizer_state = None

    if args.resume:
        LOGGER.info("Resuming from checkpoint: %s", args.resume)
        detection_model, meta = load_qat_checkpoint(detection_model, Path(args.resume))
        start_epoch = meta.get("epoch", 0) + 1
        optimizer_state = meta.get("optimizer", None)
        LOGGER.info("Resuming from epoch %d", start_epoch)
    else:
        LOGGER.info("Running PTQ calibration (%d batches)...", calib_batches)
        quant_cfg = build_quant_config(
            exclude_head=exclude_head and not args.full_int8,
            exclude_stem=exclude_stem,
        )
        detection_model = calibrate_model(
            detection_model, train_loader, quant_cfg, num_batches=calib_batches
        )
        if args.full_int8 and exclude_head:
            # Calibration ran with head excluded; now re-enable for QAT
            LOGGER.info("Full-INT8 mode: re-enabling head quantizers for QAT.")
            enable_head_quantizers(detection_model)

    # ModelOpt's mtq.quantize / calibration may move the model to CPU internally
    # and lazily initialize detection_model.criterion there, leaving criterion.proj
    # on CPU.  Restore model to target device and drop any stale criterion so it
    # re-initializes on CUDA at the first real training step.
    detection_model.to(device)
    if hasattr(detection_model, "criterion"):
        del detection_model.criterion

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        detection_model.parameters(),
        lr=lr_init,
        weight_decay=weight_decay,
    )
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
            LOGGER.info("Optimizer state restored from checkpoint.")
        except Exception as e:
            LOGGER.warning("Could not restore optimizer state: %s", e)

    # -----------------------------------------------------------------------
    # PTQ-only mode: epochs == 0
    # -----------------------------------------------------------------------
    if epochs == 0:
        LOGGER.info("epochs=0: PTQ calibration only, skipping QAT loop.")
        _save_qat_outputs(detection_model, ul_model, weights_out, scales_out, model_version)
        return

    # -----------------------------------------------------------------------
    # QAT training loop
    # -----------------------------------------------------------------------
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_ckpt_path: Optional[Path] = None

    LOGGER.info("Starting QAT: %d epochs, lr=%.2e → %.2e", epochs, lr_init, lr_min)

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

        # Validation
        v_loss = val_loss(detection_model, val_loader, device, max_batches=val_batches)
        LOGGER.info("  Val    loss=%.4f", v_loss)
        metrics["val_loss"] = v_loss

        # Checkpoint
        ckpt_path = ckpt_dir / f"qat_{model_version}_epoch{epoch + 1:03d}.modelopt"
        save_qat_checkpoint(detection_model, optimizer, epoch, metrics, ckpt_path)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_ckpt_path = ckpt_path
            # Also save the clean weights directly so we don't need mto.restore() later
            _save_qat_outputs(detection_model, ul_model, weights_out, scales_out, model_version)
            LOGGER.info("  New best val loss: %.4f  → weights/scales updated", best_val_loss)

    LOGGER.info("Training complete.  Best weights saved from epoch with val_loss=%.4f", best_val_loss)
    LOGGER.info("Output: %s  |  %s", weights_out, scales_out)


if __name__ == "__main__":
    main(sys.argv[1:])
