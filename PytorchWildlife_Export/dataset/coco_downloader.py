"""
coco_downloader.py
------------------
Downloads a filtered subset of COCO 2017 (train split) covering only the
classes needed to fill the person and vehicle gaps not covered by WCS Camera
Traps.  Uses fiftyone for selective image + annotation download, which avoids
pulling the full ~18 GB COCO zip.

Optional dependency
    pip install fiftyone
    Only required when using this module.  All other dataset utilities work
    without fiftyone installed.

Download budget estimate
    ~1 500 person images + ~500 vehicle images ≈ 300–450 MB
    (COCO images are typically 50–200 KB each, much smaller than WCS images)

COCO → MegaDetector class mapping
    person                       → 1  person
    car, truck, bus, motorcycle  → 2  vehicle
"""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path

from .annotation_converter import fo_bbox_to_yolo, write_yolo_label_file

LOGGER = logging.getLogger(__name__)

COCO_MD_CLASS_MAP: dict[str, int] = {
    "person": 1,
    "car": 2,
    "truck": 2,
    "bus": 2,
    "motorcycle": 2,
}

COCO_TARGET_CLASSES = list(COCO_MD_CLASS_MAP.keys())


def _require_fiftyone():
    try:
        import fiftyone  # noqa: F401
        import fiftyone.zoo  # noqa: F401
    except ImportError:
        raise ImportError(
            "fiftyone is required for COCO downloading.\n"
            "Install it with:  pip install fiftyone\n"
            "Alternatively, skip COCO with --skip-coco"
        )


def download_and_convert_coco(
    output_images_dir: Path,
    output_labels_dir: Path,
    max_person: int = 1500,
    max_vehicle: int = 500,
    seed: int = 42,
    fo_dataset_name: str = "coco",
) -> list[dict]:
    """Download a filtered COCO 2017 subset via fiftyone and write YOLO labels.

    fiftyone caches downloaded images locally; subsequent calls are fast.
    Returns list of {"image_path": Path, "label_path": Path} dicts.
    """
    _require_fiftyone()
    import fiftyone as fo
    import fiftyone.zoo as foz

    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Load or download the dataset
    if fo.dataset_exists(fo_dataset_name):
        LOGGER.info("Loading existing fiftyone dataset '%s' from cache.", fo_dataset_name)
        dataset = fo.load_dataset(fo_dataset_name)
    else:
        LOGGER.info(
            "Downloading COCO 2017 (train, classes=%s, max_samples=%d) via fiftyone …",
            COCO_TARGET_CLASSES,
            max_person + max_vehicle,
        )
        LOGGER.info("Expected download size: ~300–500 MB  (images only for selected classes).")
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=COCO_TARGET_CLASSES,
            max_samples=max_person + max_vehicle,
            dataset_name=fo_dataset_name,
        )

    LOGGER.info("fiftyone COCO dataset loaded: %d samples.", len(dataset))

    # Split into person-primary and vehicle-primary
    person_samples = []
    vehicle_samples = []

    for sample in dataset:
        dets = sample.ground_truth.detections if sample.ground_truth else []
        labels_in_sample = {d.label for d in dets}
        has_person = "person" in labels_in_sample
        has_vehicle = any(
            lbl in labels_in_sample for lbl in ("car", "truck", "bus", "motorcycle")
        )
        if has_person:
            person_samples.append(sample)
        if has_vehicle and not has_person:
            vehicle_samples.append(sample)

    rng = random.Random(seed)
    rng.shuffle(person_samples)
    rng.shuffle(vehicle_samples)

    selected = list(person_samples[:max_person]) + list(vehicle_samples[:max_vehicle])
    n_overlap = len(set(s.id for s in person_samples[:max_person]) &
                    set(s.id for s in vehicle_samples[:max_vehicle]))
    LOGGER.info(
        "Selected %d COCO samples (%d person-primary, %d vehicle-primary, %d overlap).",
        len(selected),
        min(len(person_samples), max_person),
        min(len(vehicle_samples), max_vehicle),
        n_overlap,
    )

    records: list[dict] = []
    skipped = 0

    for sample in selected:
        dets = sample.ground_truth.detections if sample.ground_truth else []
        yolo_anns: list[tuple] = []
        for det in dets:
            cls_id = COCO_MD_CLASS_MAP.get(det.label)
            if cls_id is None:
                continue
            xc, yc, w, h = fo_bbox_to_yolo(det.bounding_box)
            yolo_anns.append((cls_id, xc, yc, w, h))

        if not yolo_anns:
            skipped += 1
            continue

        src_path = Path(sample.filepath)
        dst_image = output_images_dir / f"coco_{src_path.name}"
        if not dst_image.exists():
            shutil.copy2(src_path, dst_image)

        label_path = output_labels_dir / (dst_image.stem + ".txt")
        write_yolo_label_file(label_path, yolo_anns)

        records.append({"image_path": dst_image, "label_path": label_path})

    LOGGER.info(
        "COCO labels written: %d images  (%d skipped — no valid bbox).",
        len(records), skipped,
    )
    return records
