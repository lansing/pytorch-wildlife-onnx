"""
annotation_converter.py
------------------------
Pure-function utilities for converting bounding-box annotations from various
source formats (COCO pixel coords, fiftyone normalised coords) to the YOLO txt
format expected by Ultralytics training, and for generating the dataset YAML
config understood by Ultralytics / PW_FT_detection.

MegaDetector class scheme
    0  animal
    1  person
    2  vehicle
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import yaml

LOGGER = logging.getLogger(__name__)

MD_CLASS_NAMES: dict[int, str] = {0: "animal", 1: "person", 2: "vehicle"}
MD_CLASS_IDS: dict[str, int] = {v: k for k, v in MD_CLASS_NAMES.items()}

_PERSON_TERMS = {"person", "people", "human", "man", "woman", "child", "group"}
_VEHICLE_TERMS = {"vehicle", "car", "truck", "bus", "motorcycle", "bike", "van"}
_SKIP_TERMS = {
    "empty",
    "blank",
    "unknown",
    "unidentified",
    "setup",
    "staff",
    "test",
    "end",
    "start",
}


def wcs_category_to_md_class(category_name: str) -> int | None:
    """Map a WCS category name to a MegaDetector class index.

    Returns None for categories that should be skipped (empty frames, setup
    shots, unidentifiable images).  All wildlife species that don't match
    person or vehicle terms are mapped to MD class 0 (animal).
    """
    name = category_name.lower().strip()
    if any(t in name for t in _SKIP_TERMS):
        return None
    if any(t in name for t in _PERSON_TERMS):
        return 1
    if any(t in name for t in _VEHICLE_TERMS):
        return 2
    return 0


def build_wcs_category_map(categories: list[dict]) -> dict[int, int | None]:
    """Build {wcs_category_id: md_class_id} from the WCS COCO categories list.

    category_id → None means "skip this category".
    """
    return {cat["id"]: wcs_category_to_md_class(cat["name"]) for cat in categories}


def coco_bbox_to_yolo(
    bbox: list[float], img_w: float, img_h: float
) -> tuple[float, float, float, float]:
    """Convert COCO bbox [x_min, y_min, w, h] (pixel coords) to YOLO format.

    Returns (x_center, y_center, width, height) all normalised to [0, 1].
    Values are clamped to [0, 1] to handle any annotation drift.
    """
    x_min, y_min, w, h = bbox
    x_center = float(x_min + w / 2) / img_w
    y_center = float(y_min + h / 2) / img_h
    width = float(w) / img_w
    height = float(h) / img_h
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def fo_bbox_to_yolo(bbox: list[float]) -> tuple[float, float, float, float]:
    """Convert a fiftyone Detection bounding_box to YOLO center format.

    fiftyone stores [x_top_left, y_top_left, width, height] already normalised
    to [0, 1].  YOLO wants [x_center, y_center, width, height].
    """
    x, y, w, h = bbox
    x_center = float(x + w / 2)
    y_center = float(y + h / 2)
    width = float(w)
    height = float(h)
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    return x_center, y_center, width, height


def write_yolo_label_file(
    label_path: Path, annotations: Iterable[tuple[int, float, float, float, float]]
) -> int:
    """Write a YOLO-format .txt label file.

    Each annotation is (class_id, x_center, y_center, width, height).
    Returns the number of annotation rows written.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(annotations)
    with open(label_path, "w") as f:
        for cls_id, xc, yc, w, h in rows:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    return len(rows)


def generate_dataset_yaml(
    output_dir: Path,
    class_names: dict[int, str] | None = None,
    train_path: str = "images/train",
    val_path: str = "images/val",
    test_path: str = "images/test",
    yaml_name: str = "megadetector_ft.yaml",
) -> Path:
    """Write an Ultralytics-compatible dataset YAML and return its path."""
    if class_names is None:
        class_names = MD_CLASS_NAMES
    yaml_data = {
        "path": str(output_dir.resolve()),
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": len(class_names),
        "names": {k: v for k, v in sorted(class_names.items())},
    }
    yaml_path = output_dir.parent.mkdir(parents=True, exist_ok=True) or output_dir
    yaml_path = output_dir / yaml_name
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
    LOGGER.info("Dataset YAML written to: %s", yaml_path)
    return yaml_path
