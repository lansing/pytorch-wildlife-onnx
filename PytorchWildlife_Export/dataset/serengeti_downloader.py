"""
serengeti_downloader.py
-----------------------
Downloads Snapshot Serengeti (S1-S11) bbox annotations and images from LILA.

Data source
    LILA Snapshot Serengeti
    https://lila.science/datasets/snapshot-serengeti

    Annotation ZIP:
        https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti/
            SnapshotSerengeti_v2.1_bboxes.json.zip

    Images:
        https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti/images/{file_name}

Category mapping
    Serengeti uses fine-grained mammal species.  All non-empty, non-vehicle
    species → MD class 0 (animal).  Uses wcs_category_to_md_class as the base
    mapping but forces any non-None, non-1, non-2 result to 0.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from .annotation_converter import wcs_category_to_md_class, coco_bbox_to_yolo, write_yolo_label_file
from .base_lila_downloader import (
    download_annotation_file,
    download_images_parallel,
    sample_empty_images,
)

LOGGER = logging.getLogger(__name__)

ANNOTATION_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-v-2-0/"
    "SnapshotSerengetiBboxes_20190903.json.zip"
)
IMAGE_BASE_URL = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/snapshotserengeti-unzipped"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorch_wildlife_export" / "serengeti"
ANNOTATION_FILENAME = "SnapshotSerengetiBboxes_20190903.json"


# ---------------------------------------------------------------------------
# Category map
# ---------------------------------------------------------------------------

def _serengeti_cat_to_md_class(name: str) -> int | None:
    """Map a Serengeti category name to an MD class.

    All wildlife species → 0 (animal).
    Person terms → 1 (person).
    Skip terms → None.
    There are no vehicle categories in Serengeti.
    """
    base = wcs_category_to_md_class(name)
    if base is None:
        return None
    # Collapse vehicle (2) to animal (0) — no vehicles in Serengeti
    if base == 2:
        return 0
    return base


def _build_serengeti_category_map(categories: list[dict]) -> dict[int, int | None]:
    return {cat["id"]: _serengeti_cat_to_md_class(cat["name"]) for cat in categories}


# ---------------------------------------------------------------------------
# Annotation download
# ---------------------------------------------------------------------------

def _download_serengeti_annotations(cache_dir: Path) -> Path:
    cache_dir = Path(cache_dir)
    json_path = cache_dir / ANNOTATION_FILENAME
    try:
        return download_annotation_file(
            url=ANNOTATION_URL,
            cache_path=json_path,
            description="Serengeti",
        )
    except RuntimeError as exc:
        LOGGER.error(
            "Failed to download Serengeti annotations.\n%s\n"
            "Check https://lila.science/datasets/snapshot-serengeti for the current URL.\n"
            "Serengeti will be skipped.",
            exc,
        )
        raise


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def _serengeti_url(file_name: str) -> str:
    return f"{IMAGE_BASE_URL}/{file_name}"


def _serengeti_local_name(source: str, file_name: str) -> str:
    # Save flat, prefixed with source to avoid collisions
    stem = Path(file_name).stem
    return f"ss_{stem}.jpg"


# ---------------------------------------------------------------------------
# Label builder
# ---------------------------------------------------------------------------

def _build_label_files(
    image_records: list[dict],
    annotations: list[dict],
    id_to_local_path: dict,
    labels_dir: Path,
    cat_map: dict[int, int | None],
) -> list[dict]:
    """Write YOLO .txt files and return records."""
    labels_dir.mkdir(parents=True, exist_ok=True)

    anns_by_img: dict = {}
    for ann in annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    records: list[dict] = []
    skipped_not_downloaded = 0
    skipped_no_valid_bbox = 0

    for img in image_records:
        img_id = img["id"]
        local_path = id_to_local_path.get(img_id)
        if local_path is None:
            skipped_not_downloaded += 1
            continue

        img_w = img.get("width", 0)
        img_h = img.get("height", 0)
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

        location = img.get("location")
        location_id = f"ss:{location}" if location is not None else None

        records.append(
            {
                "image_path": local_path,
                "label_path": label_path,
                "source": "ss",
                "location_id": location_id,
                "empty": False,
            }
        )

    LOGGER.info(
        "Serengeti labels written: %d images  (%d skipped — not downloaded, %d skipped — no valid bbox)",
        len(records), skipped_not_downloaded, skipped_no_valid_bbox,
    )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_and_convert_serengeti(
    output_dir: Path,
    max_animal: int = 700,
    max_empty: int = 200,
    seed: int = 42,
    num_workers: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[dict]:
    """Full Serengeti pipeline: download → parse → sample → download images → labels.

    Returns list of records with keys:
        image_path, label_path, source="ss", location_id, empty
    """
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        annotation_path = _download_serengeti_annotations(cache_dir)
    except RuntimeError:
        return []

    LOGGER.info("Parsing Serengeti annotation JSON: %s", annotation_path)
    with open(annotation_path) as f:
        data = json.load(f)

    cat_map = _build_serengeti_category_map(data["categories"])

    # Index annotations by image_id
    anns_by_img: dict = {}
    for ann in data["annotations"]:
        cls = cat_map.get(ann["category_id"])
        if cls is not None:
            anns_by_img.setdefault(ann["image_id"], []).append(cls)

    # Collect animal-primary image ids
    animal_img_ids = [
        img_id for img_id, classes in anns_by_img.items() if any(c == 0 for c in classes)
    ]
    LOGGER.info(
        "Serengeti images with ≥1 animal annotation: %d  (requesting %d)",
        len(animal_img_ids), max_animal,
    )

    rng = random.Random(seed)
    rng.shuffle(animal_img_ids)
    selected_ids = set(animal_img_ids[:max_animal])

    img_by_id = {img["id"]: img for img in data["images"]}
    selected_images = [img_by_id[i] for i in selected_ids if i in img_by_id]

    # Build download records
    dl_records = []
    for img in selected_images:
        file_name = img["file_name"]
        local_name = _serengeti_local_name("ss", file_name)
        dl_records.append({
            "id": img["id"],
            "file_name": file_name,
            "_local_name": local_name,
            "_url": _serengeti_url(file_name),
        })

    def url_fn(r: dict) -> str:
        return r["_url"]

    id_to_local_path = download_images_parallel(
        dl_records, images_dir, url_fn, num_workers
    )

    records = _build_label_files(
        selected_images, data["annotations"], id_to_local_path, labels_dir, cat_map
    )

    annotated_ids = selected_ids

    if max_empty > 0:
        empty_imgs = sample_empty_images(data["images"], annotated_ids, max_empty, seed)
        LOGGER.info("Sampling %d empty Serengeti images.", len(empty_imgs))

        empty_dl = []
        for img in empty_imgs:
            file_name = img["file_name"]
            local_name = _serengeti_local_name("ss", file_name)
            empty_dl.append({
                "id": img["id"],
                "_local_name": local_name,
                "_url": _serengeti_url(file_name),
                "_location": img.get("location"),
            })

        empty_id_to_path = download_images_parallel(
            empty_dl, images_dir, url_fn, num_workers
        )

        labels_dir.mkdir(parents=True, exist_ok=True)
        for r in empty_dl:
            img_id = r["id"]
            local_path = empty_id_to_path.get(img_id)
            if local_path is None:
                continue
            label_path = labels_dir / (local_path.stem + ".txt")
            label_path.touch()
            location = r.get("_location")
            location_id = f"ss:{location}" if location is not None else None
            records.append(
                {
                    "image_path": local_path,
                    "label_path": label_path,
                    "source": "ss",
                    "location_id": location_id,
                    "empty": True,
                }
            )

    LOGGER.info("Serengeti dataset ready: %d records.", len(records))
    return records
