"""
eval.py
-------
Evaluate an exported ONNX or TensorRT engine against a YOLO-format dataset
split and report per-class AP50, AP50-95, and mAP.

Supports
    • ONNX (.onnx)   — via onnxruntime; no GPU required (CUDA EP used if available)
    • TRT  (.engine) — via tensorrt + torch CUDA buffers; requires NVIDIA GPU + TRT

Usage (CLI)
-----------
    # ONNX:
    python -m PytorchWildlife_Export.dataset.eval \\
        exported_models/model.onnx \\
        --dataset data/md_ft/megadetector_ft.yaml \\
        --split val

    # TRT engine:
    python -m PytorchWildlife_Export.dataset.eval \\
        exported_models/model.engine \\
        --dataset data/md_ft/megadetector_ft.yaml \\
        --split val

Notes
-----
• Inference is run at confidence_threshold=0.001 (collect virtually all boxes)
  so the precision–recall curve spans the full range for mAP computation.
  The threshold you set only affects the mAP computation indirectly — a higher
  threshold truncates the high-recall end of the PR curve and can lower AP.
• mAP is computed using the COCO-style 101-point interpolation at IoU thresholds
  0.50 : 0.05 : 0.95 for AP50-95, and at IoU 0.50 for AP50.
• Ground-truth labels are loaded from the YOLO .txt files that accompany images
  in the dataset split (labels/ mirror of images/).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)

MD_CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}
NUM_CLASSES = 3


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def _load_gt_from_label(
    label_path: Path,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a YOLO .txt label file and return boxes (N,4) xyxy pixel + classes (N,).

    Returns empty arrays if the file is missing or has no annotations.
    """
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    rows = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            # Convert normalised xywh → pixel xyxy
            x1 = (xc - w / 2) * img_w
            y1 = (yc - h / 2) * img_h
            x2 = (xc + w / 2) * img_w
            y2 = (yc + h / 2) * img_h
            rows.append((cls, x1, y1, x2, y2))

    if not rows:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int32)

    arr = np.array(rows, dtype=np.float32)
    return arr[:, 1:], arr[:, 0].astype(np.int32)


def _label_path_for_image(image_path: Path, dataset_root: Path) -> Path:
    """Derive the YOLO .txt label path from the image path.

    Assumes the YOLO layout:
        {root}/images/{split}/{stem}.jpg  →  {root}/labels/{split}/{stem}.txt
    """
    rel = image_path.relative_to(dataset_root)
    # rel = images/{split}/{filename}
    parts = list(rel.parts)
    if parts[0] == "images":
        parts[0] = "labels"
    label_rel = Path(*parts).with_suffix(".txt")
    return dataset_root / label_rel


# ---------------------------------------------------------------------------
# IoU + detection matching
# ---------------------------------------------------------------------------

def _box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between (N,4) and (M,4) xyxy boxes → (N,M)."""
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter   = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0)


def _match_detections(
    pred_boxes: np.ndarray,   # (N, 4) xyxy pixel
    pred_scores: np.ndarray,  # (N,)
    pred_cls: np.ndarray,     # (N,) int
    gt_boxes: np.ndarray,     # (M, 4) xyxy pixel
    gt_cls: np.ndarray,       # (M,) int
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy match sorted predictions to ground-truth boxes.

    Returns (tp, fp) bool arrays of shape (N,).
    """
    n = len(pred_scores)
    tp = np.zeros(n, dtype=bool)
    fp = np.zeros(n, dtype=bool)

    if n == 0:
        return tp, fp

    order = np.argsort(-pred_scores)
    pred_boxes  = pred_boxes[order]
    pred_scores = pred_scores[order]
    pred_cls    = pred_cls[order]

    matched_gt = set()

    if len(gt_boxes) == 0:
        fp[:] = True
        # unsort
        unorder = np.argsort(order)
        return tp[unorder], fp[unorder]

    ious = _box_iou(pred_boxes, gt_boxes)  # (N, M)

    for i in range(n):
        # Consider only GT boxes of the same class
        same_cls = (gt_cls == pred_cls[i]).nonzero()[0]
        if len(same_cls) == 0:
            fp[i] = True
            continue
        iou_row = ious[i, same_cls]
        best_j  = same_cls[np.argmax(iou_row)]
        if iou_row.max() >= iou_threshold and best_j not in matched_gt:
            tp[i] = True
            matched_gt.add(best_j)
        else:
            fp[i] = True

    unorder = np.argsort(order)
    return tp[unorder], fp[unorder]


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def _compute_ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO-style 101-point interpolated AP."""
    thresholds = np.linspace(0, 1, 101)
    ap = 0.0
    for thr in thresholds:
        p = precision[recall >= thr]
        ap += p.max() if len(p) > 0 else 0.0
    return ap / 101.0


def _per_class_ap_ar(
    all_preds: list[dict],   # list per image: {boxes, scores, cls}
    all_gts: list[dict],     # list per image: {boxes, cls}
    class_id: int,
    iou_threshold: float,
) -> tuple[float, float, int]:
    """Compute AP and max-recall for a single class at a single IoU threshold.

    Returns (AP, max_recall, n_gt_instances).
    max_recall is the highest recall achievable over all confidence thresholds
    (i.e. the endpoint of the PR curve).
    """
    tp_list, fp_list, score_list = [], [], []
    n_gt = 0

    for preds, gts in zip(all_preds, all_gts):
        p_mask = preds["cls"] == class_id
        g_mask = gts["cls"]  == class_id

        p_boxes  = preds["boxes"][p_mask]
        p_scores = preds["scores"][p_mask]
        g_boxes  = gts["boxes"][g_mask]
        g_cls    = gts["cls"][g_mask]

        n_gt += int(g_mask.sum())

        if len(p_scores) == 0:
            continue

        tp, fp = _match_detections(
            p_boxes, p_scores,
            preds["cls"][p_mask],
            g_boxes, g_cls,
            iou_threshold,
        )
        tp_list.append(tp)
        fp_list.append(fp)
        score_list.append(p_scores)

    if n_gt == 0 or not score_list:
        return 0.0, 0.0, n_gt

    all_scores = np.concatenate(score_list)
    all_tp     = np.concatenate(tp_list)
    all_fp     = np.concatenate(fp_list)

    order = np.argsort(-all_scores)
    cum_tp = np.cumsum(all_tp[order]).astype(np.float64)
    cum_fp = np.cumsum(all_fp[order]).astype(np.float64)

    recall    = cum_tp / max(n_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    max_recall = float(recall[-1])
    ap = _compute_ap_101(recall, precision)
    return ap, max_recall, n_gt


def compute_map(
    all_preds: list[dict],
    all_gts: list[dict],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """Compute mAP@50, mAP@50-95, AR@50, and AR@50-95 per class.

    Parameters
    ----------
    all_preds : one dict per image with keys boxes (N,4), scores (N,), cls (N,)
    all_gts   : one dict per image with keys boxes (M,4), cls (M,)

    Returns
    -------
    {
        "mAP50": float,
        "mAP50_95": float,
        "mAR50": float,
        "mAR50_95": float,
        "per_class": {
            "animal":  {"AP50": float, "AP50_95": float,
                        "AR50": float, "AR50_95": float, "n_gt": int},
            ...
        },
    }
    AR50 / AR50_95 are the maximum recall achievable (at any confidence
    threshold), averaged over IoU thresholds for the *50-95 variant.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]

    per_class_results: dict[str, dict] = {}
    map50_values:   list[float] = []
    map50_95_values: list[float] = []
    mar50_values:   list[float] = []
    mar50_95_values: list[float] = []

    for cls_id, cls_name in MD_CLASS_NAMES.items():
        ap50, ar50, n_gt = _per_class_ap_ar(all_preds, all_gts, cls_id, iou_threshold=0.50)

        ap_at_iou: list[float] = []
        ar_at_iou: list[float] = []
        for thr in iou_thresholds:
            ap, ar, _ = _per_class_ap_ar(all_preds, all_gts, cls_id, iou_threshold=thr)
            ap_at_iou.append(ap)
            ar_at_iou.append(ar)

        ap50_95 = float(np.mean(ap_at_iou)) if ap_at_iou else 0.0
        ar50_95 = float(np.mean(ar_at_iou)) if ar_at_iou else 0.0

        per_class_results[cls_name] = {
            "AP50":    float(ap50),
            "AP50_95": ap50_95,
            "AR50":    ar50,
            "AR50_95": ar50_95,
            "n_gt":    n_gt,
        }
        map50_values.append(float(ap50))
        map50_95_values.append(ap50_95)
        mar50_values.append(ar50)
        mar50_95_values.append(ar50_95)

    return {
        "mAP50":    float(np.mean(map50_values)),
        "mAP50_95": float(np.mean(map50_95_values)),
        "mAR50":    float(np.mean(mar50_values)),
        "mAR50_95": float(np.mean(mar50_95_values)),
        "per_class": per_class_results,
    }


# ---------------------------------------------------------------------------
# Inference backends
# ---------------------------------------------------------------------------

def _load_onnx_session(model_path: str, preferred_provider: str | None):
    """Return (session, input_name, input_shape, output_name, tensor_format)."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if preferred_provider and preferred_provider in available:
        providers = available[available.index(preferred_provider):]
    else:
        providers = available

    opts = ort.SessionOptions()
    opts.enable_profiling = False
    opts.log_severity_level = 3  # suppress INFO/WARNING noise

    session = ort.InferenceSession(model_path, providers=providers, sess_options=opts)
    input_meta  = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]

    input_shape = input_meta.shape   # e.g. [1, 3, 640, 640]
    tensor_format = "nhwc" if input_shape[1] > input_shape[3] else "nchw"

    LOGGER.info(
        "ONNX session ready  providers=%s  input=%s %s  format=%s",
        [p for p in session.get_providers()],
        input_meta.name, input_shape, tensor_format,
    )
    return session, input_meta.name, input_shape, output_meta.name, tensor_format


def _run_onnx(
    session: Any,
    input_name: str,
    input_shape: list,
    output_name: str,
    tensor_format: str,
    image_path: str,
) -> tuple[np.ndarray, tuple, tuple]:
    """Run ONNX inference.  Returns (raw_output, original_dims, ratio_pad)."""
    from PytorchWildlife_Export.inference_utils.onnx_inference import preprocess_image

    uint8 = "uint8" in str(
        session.get_inputs()[0].type
    )
    preprocessed, original_dims, ratio_pad = preprocess_image(
        image_path, input_shape,
        tensor_format=tensor_format,
        normalize=not uint8,
        uint8_input=uint8,
    )
    raw_output = session.run([output_name], {input_name: preprocessed})[0]
    return raw_output, original_dims, ratio_pad


class _TRTSession:
    """Minimal TRT engine wrapper for eval (no ORT overhead)."""

    def __init__(self, engine_path: str):
        try:
            import tensorrt as trt
            import torch
        except ImportError:
            raise ImportError(
                "tensorrt and torch are required for .engine evaluation.\n"
                "Install tensorrt or evaluate using an ONNX model instead."
            )
        trt_logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        runtime = trt.Runtime(trt_logger)
        self._engine  = runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()
        self._torch   = torch

        # Discover I/O
        self.input_name = self.output_name = None
        self.input_shape = self.output_shape = None
        for i in range(self._engine.num_io_tensors):
            name  = self._engine.get_tensor_name(i)
            mode  = self._engine.get_tensor_mode(name)
            shape = tuple(self._engine.get_tensor_shape(name))
            if mode == trt.TensorIOMode.INPUT:
                self.input_name, self.input_shape = name, list(shape)
            else:
                self.output_name, self.output_shape = name, shape

        # Detect tensor format
        s = self.input_shape
        self.tensor_format = "nhwc" if s[1] > s[3] else "nchw"

        LOGGER.info(
            "TRT engine ready  input=%s %s  output=%s %s  format=%s",
            self.input_name, self.input_shape,
            self.output_name, self.output_shape,
            self.tensor_format,
        )

    def run(self, image_path: str) -> tuple[np.ndarray, tuple, tuple]:
        from PytorchWildlife_Export.inference_utils.onnx_inference import preprocess_image
        torch = self._torch

        dtype_str = str(self._engine.get_tensor_dtype(self.input_name))
        uint8 = "uint8" in dtype_str.lower() or "int8" in dtype_str.lower()

        preprocessed, original_dims, ratio_pad = preprocess_image(
            image_path, self.input_shape,
            tensor_format=self.tensor_format,
            normalize=not uint8,
            uint8_input=uint8,
        )

        in_tensor  = torch.from_numpy(preprocessed).contiguous().cuda()
        out_tensor = torch.zeros(self.output_shape, dtype=torch.float32, device="cuda")

        self._context.set_tensor_address(self.input_name,  in_tensor.data_ptr())
        self._context.set_tensor_address(self.output_name, out_tensor.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        self._context.execute_async_v3(stream_handle=stream)
        torch.cuda.synchronize()

        raw_output = out_tensor.cpu().numpy()
        return raw_output, original_dims, ratio_pad


# ---------------------------------------------------------------------------
# Postprocessing helper
# ---------------------------------------------------------------------------

def _postprocess_to_arrays(
    raw_output: np.ndarray,
    original_dims: tuple,
    input_shape: list,
    ratio_pad: tuple,
    confidence_threshold: float,
    tensor_format: str,
) -> dict:
    """Convert raw model output to {boxes, scores, cls} arrays.

    Works for YOLOv10 NMS-ready output format (1, 300, 6).
    """
    from PytorchWildlife_Export.postprocessors.util import scale_boxes_np

    det = raw_output[0]  # (300, 6): x1 y1 x2 y2 conf cls

    scores  = det[:, 4]
    keep    = scores > confidence_threshold
    det     = det[keep]

    if len(det) == 0:
        return {
            "boxes":  np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,),   dtype=np.float32),
            "cls":    np.zeros((0,),   dtype=np.int32),
        }

    boxes_raw = det[:, :4].copy()

    # Determine letterboxed input H, W accounting for tensor format
    if tensor_format == "nhwc":
        input_h, input_w = input_shape[1], input_shape[2]
    else:
        input_h, input_w = input_shape[2], input_shape[3]

    orig_h, orig_w = original_dims[1], original_dims[0]

    scaled = scale_boxes_np(
        img1_shape=(input_h, input_w),
        boxes=boxes_raw,
        img0_shape=(orig_h, orig_w),
        ratio_pad=ratio_pad,
        padding=True,
    )

    return {
        "boxes":  scaled.astype(np.float32),
        "scores": det[:, 4].astype(np.float32),
        "cls":    det[:, 5].astype(np.int32),
    }


# ---------------------------------------------------------------------------
# Main eval function
# ---------------------------------------------------------------------------

def run_eval(
    model_path: str,
    dataset_yaml: str,
    split: str = "val",
    confidence_threshold: float = 0.001,
    iou_threshold: float = 0.45,
    iou_eval_thresholds: list[float] | None = None,
    preferred_provider: str | None = None,
    max_images: int | None = None,
    quiet: bool = False,
) -> dict:
    """Evaluate an ONNX or TRT engine against a YOLO dataset split.

    Parameters
    ----------
    model_path:
        Path to .onnx or .engine file.
    dataset_yaml:
        Path to the Ultralytics YAML (e.g. megadetector_ft.yaml).
    split:
        Dataset split to evaluate: "train", "val", or "test".
    confidence_threshold:
        Detections below this score are discarded before mAP sweep.
        Use 0.001 to retain enough boxes for a meaningful PR curve.
    iou_threshold:
        IoU threshold used for TP/FP matching in the mAP computation.
        (NMS is already applied by the YOLOv10 model — this is eval-only.)
    iou_eval_thresholds:
        List of IoU thresholds for AP50-95.  Defaults to 0.50:0.05:0.95.
    preferred_provider:
        ORT execution provider, e.g. "CUDAExecutionProvider".  Ignored for TRT.
    max_images:
        Evaluate only the first N images (useful for quick smoke tests).

    Returns
    -------
    dict with keys: mAP50, mAP50_95, mAR50, mAR50_95, per_class, n_images, model_path, split.
    """
    model_path = str(model_path)
    dataset_yaml = str(dataset_yaml)

    # ------------------------------------------------------------------
    # Load dataset config + image list
    # ------------------------------------------------------------------
    with open(dataset_yaml) as f:
        ds_cfg = yaml.safe_load(f)

    dataset_root = Path(ds_cfg["path"])
    split_rel    = ds_cfg.get(split)
    if split_rel is None:
        raise ValueError(f"Split '{split}' not found in {dataset_yaml}")

    images_dir = dataset_root / split_rel
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    )
    if max_images:
        image_paths = image_paths[:max_images]

    LOGGER.info(
        "Evaluating %d images from %s split …", len(image_paths), split
    )

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    is_trt = model_path.endswith(".engine")

    if is_trt:
        trt_sess = _TRTSession(model_path)
        input_shape   = trt_sess.input_shape
        tensor_format = trt_sess.tensor_format
    else:
        ort_session, input_name, input_shape, output_name, tensor_format = (
            _load_onnx_session(model_path, preferred_provider)
        )

    # ------------------------------------------------------------------
    # Run inference
    # ------------------------------------------------------------------
    all_preds: list[dict] = []
    all_gts:   list[dict] = []

    for i, img_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            LOGGER.info("  [%d/%d] images processed …", i + 1, len(image_paths))

        # Inference
        try:
            if is_trt:
                raw_output, original_dims, ratio_pad = trt_sess.run(str(img_path))
            else:
                raw_output, original_dims, ratio_pad = _run_onnx(
                    ort_session, input_name, input_shape, output_name,
                    tensor_format, str(img_path),
                )
        except Exception as exc:
            LOGGER.warning("Inference failed for %s: %s — skipping.", img_path.name, exc)
            continue

        preds = _postprocess_to_arrays(
            raw_output, original_dims, input_shape, ratio_pad,
            confidence_threshold, tensor_format,
        )

        # Ground truth
        orig_h, orig_w = original_dims[1], original_dims[0]
        label_path = _label_path_for_image(img_path, dataset_root)
        gt_boxes, gt_cls = _load_gt_from_label(label_path, orig_w, orig_h)

        all_preds.append(preds)
        all_gts.append({"boxes": gt_boxes, "cls": gt_cls})

    LOGGER.info("Inference complete.  Computing mAP …")

    # ------------------------------------------------------------------
    # Compute mAP
    # ------------------------------------------------------------------
    map_results = compute_map(all_preds, all_gts, iou_thresholds=iou_eval_thresholds)

    results = {
        **map_results,
        "n_images":   len(all_preds),
        "model_path": model_path,
        "split":      split,
    }

    if not quiet:
        _print_eval_table(results)
    return results


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_CSV_COLUMNS = [
    "model", "model_version", "format", "size", "runtime", "n_images",
    "animal_AP50", "animal_AP50_95", "animal_AR50", "animal_AR50_95",
    "person_AP50", "person_AP50_95", "person_AR50", "person_AR50_95",
    "vehicle_AP50", "vehicle_AP50_95", "vehicle_AR50", "vehicle_AR50_95",
    "mAP50", "mAP50_95", "mAR50", "mAR50_95",
]

_CLASS_ORDER = ["animal", "person", "vehicle"]


def _parse_model_filename(model_path: str) -> dict:
    """Extract model_version / format / size / runtime from a model filename.

    Handles names like:
      MDV6-yolov10-c_float16_640_denorm_nhwc_uint8input.engine
      MDV6-yolov10-c_pruned_float16_640_denorm_nhwc_uint8input.engine
      MDV6-yolov10-e_int8_320_denorm_nhwc_uint8input.onnx
    """
    import re
    stem = Path(model_path).stem            # drop .engine / .onnx
    suffix = Path(model_path).suffix        # .engine or .onnx

    runtime = "tensorrt" if suffix == ".engine" else "onnx"

    # format: first token that is "float16", "float32", or "int8"
    fmt_match = re.search(r"(float32|float16|int8)", stem)
    fmt = fmt_match.group(1) if fmt_match else "unknown"

    # size: first standalone integer (e.g. 640, 320, 1280)
    size_match = re.search(r"_(\d{3,4})(?:_|$)", stem)
    size = int(size_match.group(1)) if size_match else 0

    # model_version: everything before the first _float / _int / _uint / _denorm token
    ver_match = re.match(r"^(.*?)(?:_(?:float|int|uint|denorm))", stem)
    model_version = ver_match.group(1) if ver_match else stem

    return {
        "model": Path(model_path).name,
        "model_version": model_version,
        "format": fmt,
        "size": size,
        "runtime": runtime,
    }


def write_csv_row(results: dict, csv_path: str) -> None:
    """Append a single eval result row to a CSV file (creates with header if new).

    The row format matches exported_models/eval_results_val.csv so results
    from different runs can be concatenated and compared directly.
    """
    import csv

    meta = _parse_model_filename(results["model_path"])
    per_class = results.get("per_class", {})

    row: dict = {
        "model":         meta["model"],
        "model_version": meta["model_version"],
        "format":        meta["format"],
        "size":          meta["size"],
        "runtime":       meta["runtime"],
        "n_images":      results["n_images"],
        "mAP50":         round(results["mAP50"],    4),
        "mAP50_95":      round(results["mAP50_95"], 4),
        "mAR50":         round(results["mAR50"],    4),
        "mAR50_95":      round(results["mAR50_95"], 4),
    }
    for cls in _CLASS_ORDER:
        stats = per_class.get(cls, {})
        row[f"{cls}_AP50"]    = round(stats.get("AP50",    0.0), 4)
        row[f"{cls}_AP50_95"] = round(stats.get("AP50_95", 0.0), 4)
        row[f"{cls}_AR50"]    = round(stats.get("AR50",    0.0), 4)
        row[f"{cls}_AR50_95"] = round(stats.get("AR50_95", 0.0), 4)

    csv_path = Path(csv_path)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row[k] for k in _CSV_COLUMNS})

    LOGGER.info("Result appended to %s", csv_path)


# ---------------------------------------------------------------------------
# Output formatter
# ---------------------------------------------------------------------------

def _print_eval_table(results: dict) -> None:
    model_name = Path(results["model_path"]).name
    width = max(len(model_name) + 4, 68)
    sep = "=" * width
    print(f"\n{sep}")
    print(f"  Eval: {model_name}")
    print(f"  Split: {results['split']}   Images: {results['n_images']}")
    print(sep)
    print(f"  {'Class':<10}  {'AP50':>8}  {'AP50-95':>10}  {'AR50':>8}  {'AR50-95':>10}  {'GT boxes':>9}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*9}")
    for cls_name, stats in results["per_class"].items():
        print(
            f"  {cls_name:<10}  {stats['AP50']:>8.4f}  {stats['AP50_95']:>10.4f}"
            f"  {stats['AR50']:>8.4f}  {stats['AR50_95']:>10.4f}  {stats['n_gt']:>9d}"
        )
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*10}")
    print(
        f"  {'mAP/mAR':<10}  {results['mAP50']:>8.4f}  {results['mAP50_95']:>10.4f}"
        f"  {results['mAR50']:>8.4f}  {results['mAR50_95']:>10.4f}"
    )
    print(sep + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ONNX or TRT engine against a YOLO dataset split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        help="Path to .onnx or .engine model file.",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Path to megadetector_ft.yaml (or any Ultralytics dataset YAML).",
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate. (default: val)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.001,
        metavar="THRESHOLD",
        help="Confidence threshold for filtering detections before mAP sweep. "
             "(default: 0.001 — keep virtually all boxes for full PR curve)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.50,
        metavar="THRESHOLD",
        help="IoU threshold for TP/FP matching. (default: 0.50)",
    )
    parser.add_argument(
        "--provider", default=None,
        help="ORT execution provider (e.g. CUDAExecutionProvider). Ignored for TRT.",
    )
    parser.add_argument(
        "--max-images", type=int, default=None, metavar="N",
        help="Evaluate only the first N images (quick smoke test).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--csv", metavar="PATH", default=None,
        help="Append eval results as a single CSV row to PATH (created with header "
             "if it does not exist).  Format matches exported_models/eval_results_val.csv "
             "so rows from multiple runs can be compared side-by-side.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Add project root to sys.path so imports work without install
    _proj_root = str(Path(__file__).resolve().parents[2])
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

    results = run_eval(
        model_path=args.model_path,
        dataset_yaml=args.dataset,
        split=args.split,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        preferred_provider=args.provider,
        max_images=args.max_images,
    )

    if args.csv:
        write_csv_row(results, args.csv)
