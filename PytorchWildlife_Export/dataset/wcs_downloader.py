"""
wcs_downloader.py
-----------------
Downloads a budget-bounded subset of the WCS Camera Traps dataset from the
LILA Azure mirror and converts annotations to YOLO format.

Data source
    LILA WCS Camera Traps — https://lila.science/datasets/wcscameratraps
    Azure LILA mirror (no credentials required):
        https://lilawildlife.blob.core.windows.net/lila-wildlife/

Annotation format
    COCO-style JSON with 5 categories:
        0  empty   → skip
        1  animal  → MD class 0
        2  person  → MD class 1
        3  group   → MD class 1  (group of people)
        4  vehicle → MD class 2

    NOTE: Only ~52 vehicle images exist in the 2020 bbox dataset.
    --wcs-max-vehicle is therefore soft-capped at ~52.

Budget
    The annotation ZIP is ~2.6 MB (28 MB uncompressed).
    Images average ~1.0–1.5 MB each.
    At the default caps (2 500 animal + 52 vehicle unique images) expect
    roughly 3.5–5 GB of image data.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import NamedTuple

from .annotation_converter import build_wcs_category_map, coco_bbox_to_yolo, write_yolo_label_file
from .base_lila_downloader import (
    download_annotation_file,
    download_images_parallel,
    sample_empty_images,
)

LOGGER = logging.getLogger(__name__)

WCS_AZURE_BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife"
WCS_IMAGE_BASE = WCS_AZURE_BASE + "/wcs-unzipped"
WCS_ANNOTATION_ZIP = "wcs_20200403_bboxes.json.zip"
WCS_ANNOTATION_JSON = "wcs_20200403_bboxes.json"
WCS_ANNOTATION_URL = f"{WCS_AZURE_BASE}/wcs/{WCS_ANNOTATION_ZIP}"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorch_wildlife_export" / "wcs"


class WCSImageRecord(NamedTuple):
    image_id: int
    file_name: str
    width: int
    height: int


# ---------------------------------------------------------------------------
# Annotation download
# ---------------------------------------------------------------------------

def download_wcs_annotations(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Path:
    """Download and extract the WCS bbox annotation JSON to *cache_dir*."""
    cache_dir = Path(cache_dir)
    json_path = cache_dir / WCS_ANNOTATION_JSON
    if force and json_path.exists():
        json_path.unlink()
    try:
        return download_annotation_file(
            url=WCS_ANNOTATION_URL,
            cache_path=json_path,
            description="WCS",
        )
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        raise


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_wcs_images(
    annotation_path: Path,
    max_animal: int = 2500,
    max_vehicle: int = 52,
    seed: int = 42,
) -> tuple[list[WCSImageRecord], list[dict], dict[int, int | None]]:
    """Parse WCS annotations and select a budget-bounded subset of images.

    Strategy
    --------
    1. Build {category_id → md_class} map.
    2. Index annotations by image_id.
    3. Split image sets by primary class (animal vs vehicle).
    4. Sample up to max_animal animal-primary and max_vehicle vehicle-primary.

    Returns (image_records, all_annotations, cat_map).
    """
    LOGGER.info("Parsing WCS annotation JSON: %s", annotation_path)
    with open(annotation_path) as f:
        data = json.load(f)

    cat_map = build_wcs_category_map(data["categories"])
    n_animal = sum(1 for v in cat_map.values() if v == 0)
    n_person = sum(1 for v in cat_map.values() if v == 1)
    n_vehicle = sum(1 for v in cat_map.values() if v == 2)
    n_skip = sum(1 for v in cat_map.values() if v is None)
    LOGGER.info(
        "WCS categories: %d total → animal=%d  person=%d  vehicle=%d  skip=%d",
        len(cat_map), n_animal, n_person, n_vehicle, n_skip,
    )

    # Build {image_id → [md_class, ...]} and image metadata
    ann_by_img: dict[int, list[int]] = {}
    for ann in data["annotations"]:
        cls = cat_map.get(ann["category_id"])
        if cls is not None:
            ann_by_img.setdefault(ann["image_id"], []).append(cls)

    # Build sets of animal-primary and vehicle-primary images
    animal_ids: list[int] = []
    vehicle_ids: list[int] = []
    for img_id, classes in ann_by_img.items():
        has_vehicle = 2 in classes
        has_animal = 0 in classes
        if has_vehicle:
            vehicle_ids.append(img_id)
        if has_animal and not has_vehicle:
            animal_ids.append(img_id)

    LOGGER.info(
        "Images with animal annotations: %d | vehicle annotations: %d",
        len(animal_ids), len(vehicle_ids),
    )

    if len(vehicle_ids) < max_vehicle:
        LOGGER.warning(
            "Only %d vehicle images available (requested %d) — will use all available.",
            len(vehicle_ids), max_vehicle,
        )

    rng = random.Random(seed)
    rng.shuffle(animal_ids)
    rng.shuffle(vehicle_ids)

    selected_ids = set(animal_ids[:max_animal]) | set(vehicle_ids[:max_vehicle])
    LOGGER.info(
        "Selected %d unique images (%d animal-primary, %d vehicle-primary, %d overlap).",
        len(selected_ids),
        min(len(animal_ids), max_animal),
        min(len(vehicle_ids), max_vehicle),
        len(set(animal_ids[:max_animal]) & set(vehicle_ids[:max_vehicle])),
    )

    # Build image records
    img_by_id = {img["id"]: img for img in data["images"]}
    records: list[WCSImageRecord] = []
    for img_id in selected_ids:
        img = img_by_id.get(img_id)
        if img is None:
            continue
        records.append(
            WCSImageRecord(
                image_id=img["id"],
                file_name=img["file_name"],
                width=img.get("width", 0),
                height=img.get("height", 0),
            )
        )

    n_anns = sum(1 for ann in data["annotations"] if ann["image_id"] in selected_ids)
    LOGGER.info("Total annotations for selected images: %d", n_anns)

    return records, data["annotations"], cat_map


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def _wcs_url(file_name: str) -> str:
    return f"{WCS_IMAGE_BASE}/{file_name}"


def download_wcs_images(
    image_records: list[WCSImageRecord],
    images_dir: Path,
    num_workers: int = 8,
    timeout: int = 30,
) -> dict[int, Path]:
    """Download WCS images with parallel workers.

    Images are stored flat under *images_dir* as ``wcs_{image_id}_{basename}``.
    Returns {image_id: local_path} for successfully downloaded images.
    """
    # Build list of dicts for the generic downloader
    dl_records = []
    for rec in image_records:
        local_name = f"wcs_{rec.image_id}_{Path(rec.file_name).name}"
        dl_records.append({
            "id": rec.image_id,
            "file_name": rec.file_name,
            "_local_name": local_name,
            "_url": _wcs_url(rec.file_name),
        })

    def url_fn(r: dict) -> str:
        return r["_url"]

    return download_images_parallel(dl_records, images_dir, url_fn, num_workers, timeout)


# ---------------------------------------------------------------------------
# Label file builder
# ---------------------------------------------------------------------------

def build_wcs_label_files(
    image_records: list[WCSImageRecord],
    annotations: list[dict],
    id_to_local_path: dict[int, Path],
    labels_dir: Path,
    cat_map: dict[int, int | None],
) -> list[dict]:
    """Write YOLO .txt label files for downloaded WCS images.

    Returns a list of records with the new format:
        image_path, label_path, source="wcs", location_id, empty=False
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Group annotations by image_id
    anns_by_img: dict[int, list[dict]] = {}
    for ann in annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    # Build img_meta from image_records
    img_meta = {
        rec.image_id: {
            "width": rec.width,
            "height": rec.height,
        }
        for rec in image_records
    }

    records: list[dict] = []
    skipped_not_downloaded = 0
    skipped_no_valid_bbox = 0

    for rec in image_records:
        img_id = rec.image_id
        local_path = id_to_local_path.get(img_id)
        if local_path is None:
            skipped_not_downloaded += 1
            continue

        img_w = rec.width
        img_h = rec.height
        if not img_w or not img_h:
            try:
                import cv2
                arr = cv2.imread(str(local_path))
                if arr is not None:
                    img_h, img_w = arr.shape[:2]
            except Exception:
                pass

        if not img_w or not img_h:
            LOGGER.debug("Could not read dimensions for %s — skipping.", local_path)
            skipped_not_downloaded += 1
            continue

        anns = anns_by_img.get(img_id, [])
        yolo_anns: list[tuple] = []
        for ann in anns:
            cls_id = cat_map.get(ann["category_id"])
            if cls_id is None:
                continue
            bbox = ann.get("bbox")
            if not bbox:
                continue
            xc, yc, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
            yolo_anns.append((cls_id, xc, yc, w, h))

        if not yolo_anns:
            skipped_no_valid_bbox += 1
            continue

        label_path = labels_dir / (local_path.stem + ".txt")
        write_yolo_label_file(label_path, yolo_anns)

        records.append(
            {
                "image_path": local_path,
                "label_path": label_path,
                "source": "wcs",
                "location_id": None,  # filled below if available
                "empty": False,
            }
        )

    LOGGER.info(
        "WCS labels written: %d images  (%d skipped — not downloaded, %d skipped — no valid bbox)",
        len(records), skipped_not_downloaded, skipped_no_valid_bbox,
    )
    return records


# ---------------------------------------------------------------------------
# Empty image download
# ---------------------------------------------------------------------------

def _download_empty_wcs(
    annotation_path: Path,
    annotated_ids: set[int],
    images_dir: Path,
    labels_dir: Path,
    max_empty: int,
    seed: int,
    num_workers: int,
    timeout: int,
) -> list[dict]:
    """Download and build records for empty (no-annotation) WCS images."""
    with open(annotation_path) as f:
        data = json.load(f)

    all_images = data["images"]
    empty_imgs = sample_empty_images(all_images, annotated_ids, max_empty, seed)
    LOGGER.info("Sampling %d empty WCS images (unannotated).", len(empty_imgs))

    dl_records = []
    for img in empty_imgs:
        local_name = f"wcs_{img['id']}_{Path(img['file_name']).name}"
        dl_records.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "_local_name": local_name,
            "_url": _wcs_url(img["file_name"]),
            "_location": img.get("location"),
        })

    def url_fn(r: dict) -> str:
        return r["_url"]

    id_to_path = download_images_parallel(dl_records, images_dir, url_fn, num_workers, timeout)

    labels_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for r in dl_records:
        img_id = r["id"]
        local_path = id_to_path.get(img_id)
        if local_path is None:
            continue
        label_path = labels_dir / (local_path.stem + ".txt")
        # Write empty label file
        label_path.touch()
        location = r.get("_location")
        location_id = f"wcs:{location}" if location is not None else None
        records.append(
            {
                "image_path": local_path,
                "label_path": label_path,
                "source": "wcs",
                "location_id": location_id,
                "empty": True,
            }
        )

    LOGGER.info("WCS empty records: %d", len(records))
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_and_convert_wcs(
    output_images_dir: Path,
    output_labels_dir: Path,
    max_animal: int = 2500,
    max_vehicle: int = 52,
    max_empty: int = 0,
    seed: int = 42,
    num_workers: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_annotation_download: bool = False,
) -> list[dict]:
    """Full WCS pipeline: download → parse → sample → download images → labels.

    Returns list of records with keys:
        image_path, label_path, source, location_id, empty
    """
    annotation_path = download_wcs_annotations(cache_dir, force=force_annotation_download)

    with open(annotation_path) as f:
        data = json.load(f)

    cat_map = build_wcs_category_map(data["categories"])
    image_records, annotations, cat_map = sample_wcs_images(
        annotation_path, max_animal, max_vehicle, seed
    )

    id_to_local_path = download_wcs_images(
        image_records, output_images_dir, num_workers
    )

    records = build_wcs_label_files(
        image_records, annotations, id_to_local_path, output_labels_dir, cat_map
    )

    # Patch location_id using the raw image data
    img_location_map: dict[int, int | None] = {}
    for img in data["images"]:
        img_location_map[img["id"]] = img.get("location")

    for rec in records:
        # Find the image_id from the local path stem — it's wcs_{id}_{basename}
        stem = rec["image_path"].stem
        parts = stem.split("_", 2)
        if len(parts) >= 2 and parts[0] == "wcs":
            try:
                img_id = int(parts[1])
                location = img_location_map.get(img_id)
                rec["location_id"] = f"wcs:{location}" if location is not None else None
            except ValueError:
                pass

    annotated_ids = {rec.image_id for rec in image_records}

    if max_empty > 0:
        empty_records = _download_empty_wcs(
            annotation_path,
            annotated_ids,
            output_images_dir,
            output_labels_dir,
            max_empty,
            seed,
            num_workers,
            30,
        )
        records.extend(empty_records)

    return records
