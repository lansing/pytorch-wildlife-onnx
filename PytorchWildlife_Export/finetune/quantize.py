"""
quantize.py
-----------
ModelOpt INT8 quantization setup for MDV6-yolov10.

Handles:
  - Building the selective quantization config (head / stem exclusions)
  - Running the PTQ calibration forward pass
  - Saving / loading ModelOpt quantized state
  - Extracting QAT-learned scales for use in our own QDQ export pipeline

After QAT training the workflow is:
  1. extract_qat_scales()      → save scales JSON
  2. extract_clean_state_dict() → save finetuned float32 .pt
  3. export_tool.py --weights <.pt> --scales-json <.json> --format int8
     → our own QDQ graph is built with QAT-learned activation scales,
       complemented by ORT calibration for output tensor scales.

Requires: pip install "nvidia-modelopt[torch]>=0.21"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def _require_modelopt():
    try:
        import modelopt.torch.quantization as mtq
        import modelopt.torch.opt as mto
        return mtq, mto
    except ImportError:
        raise ImportError(
            "nvidia-modelopt is required for QAT.\n"
            "Install with:  pip install 'nvidia-modelopt[torch]>=0.21'\n"
            "Inside Docker: pip install 'nvidia-modelopt[torch]>=0.21' -q"
        )


def build_quant_config(
    exclude_head: bool = True,
    exclude_stem: bool = True,
) -> dict:
    """Return a ModelOpt INT8 quantization config for YOLOv10.

    Parameters
    ----------
    exclude_head : exclude model.23 (detection head) from INT8.
        Recommended for initial PTQ/QAT — the head is sensitive to scale errors.
        Set False to attempt full INT8 (requires QAT to recover accuracy).
    exclude_stem : exclude model.0 (3-channel input conv) from INT8.
        3-channel convolutions have no INT8 Tensor Core path on most hardware.
    """
    import copy
    mtq, _ = _require_modelopt()

    cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)

    if exclude_stem:
        cfg["quant_cfg"]["*model.0*"] = {"enable": False}
        LOGGER.info("Quantization: excluding stem (model.0)")

    if exclude_head:
        cfg["quant_cfg"]["*model.23*"] = {"enable": False}
        LOGGER.info("Quantization: excluding head (model.23)")
    else:
        LOGGER.info(
            "Quantization: head (model.23) INCLUDED — full INT8 mode. "
            "Expect accuracy drop unless QAT recovers it."
        )

    return cfg


def calibrate_model(
    model: nn.Module,
    calib_loader: torch.utils.data.DataLoader,
    quant_cfg: dict,
    num_batches: int = 8,
) -> nn.Module:
    """Apply PTQ calibration via ModelOpt mtq.quantize().

    Runs *num_batches* forward passes (no gradients, no labels) to compute
    activation statistics for the initial INT8 scales.

    Parameters
    ----------
    model       : DetectionModel (ul_model.model) in eval mode on CUDA
    calib_loader: DataLoader; only 'img' tensors are used
    quant_cfg   : config from build_quant_config()
    num_batches : how many batches to use for calibration (100–200 images typical)

    Returns
    -------
    The same model object, now wrapped with fake-quantization modules.
    """
    mtq, _ = _require_modelopt()

    seen = 0

    def _calibrate_fn(m: nn.Module) -> None:
        nonlocal seen
        m.eval()
        with torch.no_grad():
            for batch in calib_loader:
                imgs = batch["img"].cuda().float() / 255.0
                m(imgs)
                seen += len(imgs)
                if seen >= num_batches * calib_loader.batch_size:
                    break

    LOGGER.info(
        "PTQ calibration: ~%d images  (%d batches × %d)",
        num_batches * calib_loader.batch_size,
        num_batches, calib_loader.batch_size,
    )
    model = mtq.quantize(model, quant_cfg, _calibrate_fn)
    mtq.print_quant_summary(model)

    # Ultralytics loads models for inference with requires_grad=False.
    # Re-enable gradients for all float parameters so QAT backprop works.
    # TensorQuantizer amax buffers are non-floating-point or non-leaf — skip them.
    n_trainable = 0
    for name, p in model.named_parameters():
        if p.is_floating_point() and type(p).__name__ != "TensorQuantizer":
            p.requires_grad_(True)
            n_trainable += 1
    LOGGER.info("QAT: %d floating-point parameters set to requires_grad=True", n_trainable)
    model.train()
    return model


def enable_head_quantizers(model: nn.Module) -> None:
    """Re-enable model.23 quantizers post-calibration (for full-INT8 QAT experiments).

    Call this after PTQ calibration when you want to bring the head into INT8
    via QAT gradient updates.
    """
    mtq, _ = _require_modelopt()
    mtq.enable_quantizer(model, "*model.23*")
    LOGGER.info("Head quantizers (model.23) enabled for QAT.")


def extract_qat_scales(model: nn.Module) -> dict:
    """Extract QAT-learned INT8 scales from a ModelOpt-quantized model.

    Returns a JSON-serializable dict:
        {
            "/model.1/cv1/conv/Conv": {
                "input_scales": [0.0234],   # activation input scale(s)
                "weight_scale": 0.0156,     # weight scale (per-tensor max)
            },
            ...
        }

    The ONNX node name is derived from the PyTorch module path using Ultralytics'
    ONNX export naming convention: replace '.' with '/', prepend '/', append
    '/{op_type}'.  This matches the node names in the ONNX produced by our
    export pipeline.

    Notes
    -----
    ModelOpt INT8_DEFAULT_CFG places input_quantizer and weight_quantizer on
    each Conv/Linear but does NOT place output_quantizer.  Output tensor scales
    are therefore not available here — our export pipeline fills them via ORT
    calibration on the finetuned float32 ONNX.  The QAT-learned input_scales
    override the ORT-calibrated activation scales (they are gradient-optimized
    and therefore more accurate).
    """
    # Identify TensorQuantizer by class name — handles API changes across ModelOpt versions
    # (the import path for TensorQuantizer has moved between 0.2x and 0.4x releases).
    def _is_tq(m) -> bool:
        return type(m).__name__ == "TensorQuantizer"

    # Walk named modules and collect amax values per quantizer role
    parent_data: dict[str, dict] = {}   # {parent_path: {role: float}}
    for name, module in model.named_modules():
        if not _is_tq(module):
            continue
        if not module.is_enabled:
            continue
        if module.amax is None:
            LOGGER.debug("TensorQuantizer %s has no amax — not calibrated, skipping", name)
            continue

        amax_val = float(module.amax.max().item())
        scale = amax_val / 127.0

        if name.endswith(".input_quantizer"):
            parent = name[: -len(".input_quantizer")]
            role = "input"
        elif name.endswith(".weight_quantizer"):
            parent = name[: -len(".weight_quantizer")]
            role = "weight"
        elif name.endswith(".output_quantizer"):
            parent = name[: -len(".output_quantizer")]
            role = "output"
        else:
            LOGGER.debug("Unknown quantizer suffix in %s, skipping", name)
            continue

        parent_data.setdefault(parent, {})[role] = scale

    # Convert PyTorch module paths → ONNX node names
    module_map = dict(model.named_modules())
    result: dict = {}

    for parent_path, roles in parent_data.items():
        parent_mod = module_map.get(parent_path)
        if parent_mod is None:
            LOGGER.warning("Module not found at path %s — skipping scale extraction", parent_path)
            continue

        if isinstance(parent_mod, nn.Conv2d):
            op_suffix = "Conv"
        elif isinstance(parent_mod, nn.Linear):
            op_suffix = "Gemm"
        else:
            op_suffix = "Conv"

        onnx_name = "/" + parent_path.replace(".", "/") + "/" + op_suffix
        entry: dict = {}
        if "input" in roles:
            # input_scales is a list matching the number of activation inputs
            # (Conv has 1 activation input; MatMul has 2)
            entry["input_scales"] = [roles["input"]]
        if "weight" in roles:
            entry["weight_scale"] = roles["weight"]
        if "output" in roles:
            entry["output_scale"] = roles["output"]
        result[onnx_name] = entry

    LOGGER.info("Extracted QAT scales for %d quantized layers", len(result))
    return result


def extract_clean_state_dict(model: nn.Module) -> dict:
    """Return state_dict with ModelOpt TensorQuantizer parameters removed.

    The returned dict can be loaded into a standard (non-quantized) Ultralytics
    DetectionModel with strict=False, giving a clean float32 checkpoint that
    carries the QAT-finetuned weights without any ModelOpt overhead.
    """
    quantizer_paths: set[str] = set()
    for name, mod in model.named_modules():
        if type(mod).__name__ == "TensorQuantizer":
            quantizer_paths.add(name)

    clean: dict = {}
    for key, val in model.state_dict().items():
        parts = key.split(".")
        is_quantizer = any(
            ".".join(parts[:i]) in quantizer_paths
            for i in range(1, len(parts) + 1)
        )
        if not is_quantizer:
            clean[key] = val

    LOGGER.info(
        "Clean state_dict: %d / %d keys retained (quantizer params stripped)",
        len(clean), len(model.state_dict()),
    )
    return clean


def save_qat_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    """Save a ModelOpt-compatible checkpoint (weights + quantizer state)."""
    _, mto = _require_modelopt()
    path.parent.mkdir(parents=True, exist_ok=True)
    mto.save(model, str(path))
    meta_path = path.with_suffix(".meta.pt")
    torch.save({
        "epoch":     epoch,
        "metrics":   metrics,
        "optimizer": optimizer.state_dict(),
    }, meta_path)
    LOGGER.info("QAT checkpoint saved: %s", path)


def load_qat_checkpoint(model: nn.Module, path: Path) -> tuple[nn.Module, dict]:
    """Load a ModelOpt checkpoint into *model*.  Returns (model, metadata)."""
    _, mto = _require_modelopt()
    mto.restore(model, str(path))
    meta_path = path.with_suffix(".meta.pt")
    meta = torch.load(meta_path, map_location="cpu") if meta_path.exists() else {}
    LOGGER.info("QAT checkpoint loaded: %s  (epoch %s)", path, meta.get("epoch", "?"))
    return model, meta
