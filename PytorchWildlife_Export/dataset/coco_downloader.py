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
from pathlib import Path

from .annotation_converter import fo_bbox_to_yolo, write_yolo_label_file

LOGGER = logging.getLogger(__name__)

# COCO class names → MD class index
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
        import fiftyone as fo
        import fiftyone.zoo as foz
        return fo, foz
    except ImportError:
        raise ImportError(
            "fiftyone is required for COCO downloading.\n"
            "Install it with:  pip install fiftyone\n"
            "Alternatively, skip COCO with --skip-coco and provide person/vehicle "
            "images manually."
        )


def download_and_convert_coco(
    output_images_dir: Path,
    output_labels_dir: Path,
    max_person: int = 1500,
    max_vehicle: int = 500,
    seed: int = 42,
    fo_dataset_name: str = "coco2017-md-subset",
) -> list[dict]:
    """Download a filtered COCO 2017 subset via fiftyone and write YOLO labels.

    fiftyone caches downloaded images locally; a second call with the same
    *fo_dataset_name* reuses the cache without re-downloading.

    Returns list of {"image_path": Path, "label_path": Path} dicts.
    """
    fo, foz = _require_fiftyone()

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load (or download) filtered COCO 2017 train subset via fiftyone zoo #
    # ------------------------------------------------------------------ #
    # We request more samples than needed so we can subsample after filtering.
    request_samples = (max_person + max_vehicle) * 2 + 500

    if fo.dataset_exists(fo_dataset_name):
        LOGGER.info(
            "Loading existing fiftyone dataset '%s' from cache.", fo_dataset_name
        )
        dataset = fo.load_dataset(fo_dataset_name)
    else:
        LOGGER.info(
            "Downloading COCO 2017 (train, classes=%s, max_samples=%d) via fiftyone …",
            COCO_TARGET_CLASSES,
            request_samples,
        )
        LOGGER.info(
            "Expected download size: ~300–500 MB  (images only for selected classes)."
        )
        dataset = foz.load_zoo_dataset(
            "coco-2017",
            split="train",
            label_types=["detections"],
            classes=COCO_TARGET_CLASSES,
            max_samples=request_samples,
            dataset_name=fo_dataset_name,
            persistent=True,  # cache for future runs
        )
        LOGGER.info(
            "fiftyone COCO dataset loaded: %d samples.", len(dataset)
        )

    # ------------------------------------------------------------------ #
    # Separate person-primary and vehicle-primary images, then subsample  #
    # ------------------------------------------------------------------ #
    rng = random.Random(seed)

    person_samples = []
    vehicle_samples = []

    for sample in dataset:
        if sample.ground_truth is None:
            continue
        labels = [d.label for d in sample.ground_truth.detections]
        if "person" in labels:
            person_samples.append(sample)
        elif any(l in labels for l in ("car", "truck", "bus", "motorcycle")):
            vehicle_samples.append(sample)

    rng.shuffle(person_samples)
    rng.shuffle(vehicle_samples)
    selected_person  = person_samples[:max_person]
    selected_vehicle = vehicle_samples[:max_vehicle]

    # Use a set to deduplicate (a sample selected for person may also have vehicles)
    selected_ids = {s.id for s in selected_person} | {s.id for s in selected_vehicle}
    selected = [s for s in dataset if s.id in selected_ids]

    LOGGER.info(
        "Selected %d COCO samples (%d person-primary, %d vehicle-primary, "
        "%d overlap).",
        len(selected),
        len({s.id for s in selected_person} - {s.id for s in selected_vehicle}),
        len({s.id for s in selected_vehicle} - {s.id for s in selected_person}),
        len({s.id for s in selected_person} & {s.id for s in selected_vehicle}),
    )

    # ------------------------------------------------------------------ #
    # Convert to YOLO format and symlink / copy images                    #
    # ------------------------------------------------------------------ #
    records_out: list[dict] = []
    skipped_no_bbox = 0

    for sample in selected:
        src_img = Path(sample.filepath)
        dest_img = output_images_dir / src_img.name
        label_path = output_labels_dir / (src_img.stem + ".txt")

        # Collect YOLO annotation rows for this sample
        yolo_rows: list[tuple[int, float, float, float, float]] = []
        if sample.ground_truth is not None:
            for det in sample.ground_truth.detections:
                md_cls = COCO_MD_CLASS_MAP.get(det.label)
                if md_cls is None:
                    continue
                xc, yc, w, h = fo_bbox_to_yolo(det.bounding_box)
                if w <= 0 or h <= 0:
                    continue
                yolo_rows.append((md_cls, xc, yc, w, h))

        if not yolo_rows:
            skipped_no_bbox += 1
            continue

        # Always copy COCO images — fiftyone's cache is container-local
        # and symlinks would break when a new container mounts the dataset.
        if not dest_img.exists():
            import shutil
            shutil.copy2(src_img, dest_img)

        write_yolo_label_file(label_path, yolo_rows)
        records_out.append({"image_path": dest_img, "label_path": label_path})

    LOGGER.info(
        "COCO labels written: %d images  (%d skipped — no valid bbox).",
        len(records_out), skipped_no_bbox,
    )
    return records_out
