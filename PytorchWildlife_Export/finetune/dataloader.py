"""
dataloader.py
-------------
Build Ultralytics-compatible DataLoaders from our YOLO-format dataset.

Thin wrappers around Ultralytics' own build_yolo_dataset / build_dataloader so
the rest of the training loop can stay framework-agnostic.
"""

from __future__ import annotations

import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader


def _ultralytics_args(imgsz: int, batch_size: int, workers: int, augment: bool):
    """Build a minimal Ultralytics config namespace for dataset construction."""
    from ultralytics.cfg import get_cfg, DEFAULT_CFG
    args = get_cfg(DEFAULT_CFG)
    args.imgsz  = imgsz
    args.batch  = batch_size
    args.workers = workers
    args.cache  = False
    args.rect   = False     # fixed-size batches (required for QAT)
    args.augment = augment
    args.fraction = 1.0
    args.task   = "detect"
    args.overlap_mask = True
    args.mask_ratio   = 4
    args.single_cls   = False
    args.classes      = None
    args.verbose      = False
    return args


def build_md_dataloader(
    dataset_yaml: str,
    split: str,
    imgsz: int = 640,
    batch_size: int = 8,
    workers: int = 4,
    augment: bool | None = None,
) -> DataLoader:
    """Return a DataLoader for the given dataset split.

    Parameters
    ----------
    dataset_yaml : path to megadetector_ft.yaml (or any Ultralytics YAML)
    split        : "train", "val", or "test"
    augment      : mosaic/flip augmentation; defaults to True for train, False otherwise
    """
    from ultralytics.data.build import build_yolo_dataset, build_dataloader

    if augment is None:
        augment = (split == "train")

    with open(dataset_yaml) as f:
        data_cfg = yaml.safe_load(f)

    # Resolve relative paths in the YAML
    dataset_root = Path(data_cfg["path"])
    split_rel = data_cfg.get(split)
    if split_rel is None:
        raise ValueError(f"Split '{split}' not found in {dataset_yaml}")
    img_path = str(dataset_root / split_rel)

    args = _ultralytics_args(imgsz, batch_size, workers, augment)
    dataset = build_yolo_dataset(args, img_path, batch_size, data_cfg, mode=split)
    shuffle = (split == "train")
    loader  = build_dataloader(dataset, batch=batch_size, workers=workers,
                               shuffle=shuffle, rank=-1)
    return loader


def to_device(batch: dict, device: str | torch.device) -> dict:
    """Move an Ultralytics batch dict to the target device."""
    dev = torch.device(device)
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(dev, non_blocking=True)
        else:
            out[k] = v
    return out
