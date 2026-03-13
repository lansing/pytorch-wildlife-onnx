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

import io
import json
import logging
import random
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from .annotation_converter import (
    build_wcs_category_map,
    coco_bbox_to_yolo,
    write_yolo_label_file,
)

LOGGER = logging.getLogger(__name__)

WCS_AZURE_BASE = "https://lilawildlife.blob.core.windows.net/lila-wildlife"
WCS_IMAGE_BASE = f"{WCS_AZURE_BASE}/wcs-unzipped"

# Annotation ZIP: ~2.6 MB compressed, ~28 MB uncompressed
WCS_ANNOTATION_ZIP = "wcs_20200403_bboxes.json.zip"
WCS_ANNOTATION_JSON = "wcs_20200403_bboxes.json"
WCS_ANNOTATION_URL  = f"{WCS_AZURE_BASE}/wcs/{WCS_ANNOTATION_ZIP}"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorch_wildlife_export" / "wcs"


class WCSImageRecord(NamedTuple):
    image_id: str       # UUID string (e.g. "d6d634e4-92d4-11e9-...")
    file_name: str      # relative path within WCS (e.g. "animals/0043/0766.jpg")
    width: int
    height: int


# ---------------------------------------------------------------------------
# Step 1: Download + extract annotation JSON
# ---------------------------------------------------------------------------

def download_wcs_annotations(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Path:
    """Download and extract the WCS bbox annotation JSON to *cache_dir*.

    The raw ZIP is cached; extraction is re-done on each call if the JSON is
    missing.  Returns the path to the local JSON file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / WCS_ANNOTATION_JSON
    zip_path  = cache_dir / WCS_ANNOTATION_ZIP

    if json_path.exists() and not force:
        LOGGER.info("WCS annotations already cached at: %s", json_path)
        return json_path

    # Download ZIP if not present
    if not zip_path.exists() or force:
        LOGGER.info("Downloading WCS annotation ZIP from:\n  %s", WCS_ANNOTATION_URL)
        LOGGER.info("File is ~2.6 MB — should be fast.")

        def _report(block_num: int, block_size: int, total_size: int) -> None:
            downloaded = block_num * block_size
            if total_size > 0 and block_num % 200 == 0:
                pct = min(100.0, downloaded / total_size * 100)
                LOGGER.info("  %.1f%%", pct)

        tmp_zip = zip_path.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(WCS_ANNOTATION_URL, tmp_zip, reporthook=_report)
            tmp_zip.rename(zip_path)
        except Exception as exc:
            if tmp_zip.exists():
                tmp_zip.unlink()
            raise RuntimeError(
                f"Failed to download WCS annotations from {WCS_ANNOTATION_URL}.\n"
                f"Error: {exc}\n"
                "Check https://lila.science/datasets/wcscameratraps for current URLs."
            ) from exc

        LOGGER.info("WCS annotation ZIP cached: %s", zip_path)

    # Extract JSON from ZIP
    LOGGER.info("Extracting %s …", WCS_ANNOTATION_JSON)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(WCS_ANNOTATION_JSON) as src, open(json_path, "wb") as dst:
            dst.write(src.read())
    LOGGER.info(
        "WCS annotation JSON extracted: %s  (%.1f MB)",
        json_path, json_path.stat().st_size / 1024 / 1024,
    )
    return json_path


# ---------------------------------------------------------------------------
# Step 2: Parse + sample
# ---------------------------------------------------------------------------

def sample_wcs_images(
    annotation_path: Path,
    max_animal: int = 2500,
    max_vehicle: int = 500,
    seed: int = 42,
) -> tuple[list[WCSImageRecord], list[dict]]:
    """Parse WCS annotations and select a budget-bounded subset of images.

    Strategy
    --------
    1. Build {category_id → md_class} map from the categories list.
    2. Index annotations per image (animal and vehicle bbox only).
    3. Shuffle + subsample up to max_animal animal-primary images.
    4. Shuffle + subsample up to max_vehicle vehicle-primary images.
       NOTE: only ~52 vehicle images exist — the cap is effectively ~52.
    5. Take the union so each image is counted once.

    Returns
    -------
    (image_records, filtered_annotations)
    """
    LOGGER.info("Parsing WCS annotation JSON: %s", annotation_path)
    with open(annotation_path) as f:
        data = json.load(f)

    cat_map = build_wcs_category_map(data["categories"])
    LOGGER.info(
        "WCS categories: %d total → animal=%d  person=%d  vehicle=%d  skip=%d",
        len(cat_map),
        sum(1 for v in cat_map.values() if v == 0),
        sum(1 for v in cat_map.values() if v == 1),
        sum(1 for v in cat_map.values() if v == 2),
        sum(1 for v in cat_map.values() if v is None),
    )

    image_by_id: dict[str, dict] = {img["id"]: img for img in data["images"]}

    # Index annotations per image, keeping only animal (0) and vehicle (2)
    animal_annots_by_image: dict[str, list[dict]] = {}
    vehicle_annots_by_image: dict[str, list[dict]] = {}

    for ann in data["annotations"]:
        md_cls = cat_map.get(ann.get("category_id"))
        if md_cls == 0:
            animal_annots_by_image.setdefault(ann["image_id"], []).append(ann)
        elif md_cls == 2:
            vehicle_annots_by_image.setdefault(ann["image_id"], []).append(ann)

    LOGGER.info(
        "Images with animal annotations: %d | vehicle annotations: %d",
        len(animal_annots_by_image),
        len(vehicle_annots_by_image),
    )

    if len(vehicle_annots_by_image) < max_vehicle:
        LOGGER.warning(
            "Only %d vehicle images available (requested %d) — "
            "will use all available.",
            len(vehicle_annots_by_image), max_vehicle,
        )

    rng = random.Random(seed)

    animal_ids = list(animal_annots_by_image.keys())
    rng.shuffle(animal_ids)
    selected_animal_ids = set(animal_ids[:max_animal])

    vehicle_ids = list(vehicle_annots_by_image.keys())
    rng.shuffle(vehicle_ids)
    selected_vehicle_ids = set(vehicle_ids[:max_vehicle])

    selected_ids = selected_animal_ids | selected_vehicle_ids
    LOGGER.info(
        "Selected %d unique images (%d animal-primary, %d vehicle-primary, "
        "%d overlap).",
        len(selected_ids),
        len(selected_animal_ids - selected_vehicle_ids),
        len(selected_vehicle_ids - selected_animal_ids),
        len(selected_animal_ids & selected_vehicle_ids),
    )

    image_records: list[WCSImageRecord] = []
    for img_id in selected_ids:
        img = image_by_id.get(img_id)
        if img is None:
            continue
        image_records.append(
            WCSImageRecord(
                image_id=img["id"],
                file_name=img["file_name"],
                width=img.get("width", 0),
                height=img.get("height", 0),
            )
        )

    filtered_annotations: list[dict] = []
    for img_id in selected_ids:
        filtered_annotations.extend(animal_annots_by_image.get(img_id, []))
        filtered_annotations.extend(vehicle_annots_by_image.get(img_id, []))

    LOGGER.info("Total annotations for selected images: %d", len(filtered_annotations))
    return image_records, filtered_annotations


# ---------------------------------------------------------------------------
# Step 3: Download images
# ---------------------------------------------------------------------------

def _download_one(
    record: WCSImageRecord,
    dest_path: Path,
    timeout: int,
) -> tuple[WCSImageRecord, bool]:
    """Download a single WCS image.  Returns (record, success)."""
    if dest_path.exists():
        return record, True
    url = f"{WCS_IMAGE_BASE}/{record.file_name}"
    tmp = dest_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest_path)
        return record, True
    except Exception as exc:
        LOGGER.debug("Failed to download %s: %s", url, exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return record, False


def download_wcs_images(
    image_records: list[WCSImageRecord],
    images_dir: Path,
    num_workers: int = 8,
    timeout: int = 30,
) -> dict[str, Path]:
    """Download WCS images with parallel workers.

    Images are stored flat under *images_dir* as ``{image_id}_{basename}``.
    Returns {image_id: local_path} for successfully downloaded images only.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    id_to_path: dict[str, Path] = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for rec in image_records:
            # Flatten path; prefix with image_id to guard against basename collisions
            basename = Path(rec.file_name).name
            # UUID image IDs contain hyphens which are safe in filenames
            dest = images_dir / f"{rec.image_id}_{basename}"
            id_to_path[rec.image_id] = dest
            fut = pool.submit(_download_one, rec, dest, timeout)
            futures[fut] = rec

        ok = 0
        fail = 0
        for i, fut in enumerate(as_completed(futures), 1):
            rec, success = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                id_to_path.pop(rec.image_id, None)
            if i % 100 == 0 or i == len(futures):
                LOGGER.info(
                    "  Images: %d/%d done  (%d ok, %d failed)",
                    i, len(futures), ok, fail,
                )

    LOGGER.info("WCS image download complete: %d downloaded, %d failed.", ok, fail)
    return id_to_path


# ---------------------------------------------------------------------------
# Step 4: Write YOLO labels
# ---------------------------------------------------------------------------

def build_wcs_label_files(
    image_records: list[WCSImageRecord],
    annotations: list[dict],
    id_to_local_path: dict[str, Path],
    labels_dir: Path,
    cat_map: dict[int, int | None],
) -> list[dict]:
    """Write YOLO .txt label files for downloaded WCS images.

    Returns a list of {\"image_path\": Path, \"label_path\": Path} dicts for all
    images that have at least one valid annotation and were downloaded
    successfully.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    annots_by_image: dict[str, list[dict]] = {}
    for ann in annotations:
        annots_by_image.setdefault(ann["image_id"], []).append(ann)

    img_meta: dict[str, WCSImageRecord] = {r.image_id: r for r in image_records}

    records_out: list[dict] = []
    skipped_no_download = 0
    skipped_no_bbox = 0

    for img_id, img_annots in annots_by_image.items():
        local_img = id_to_local_path.get(img_id)
        if local_img is None:
            skipped_no_download += 1
            continue

        rec = img_meta[img_id]
        img_w, img_h = rec.width, rec.height

        if img_w == 0 or img_h == 0:
            try:
                import cv2
                im = cv2.imread(str(local_img))
                if im is not None:
                    img_h, img_w = im.shape[:2]
            except Exception:
                LOGGER.debug("Could not read dimensions for %s — skipping.", local_img)
                continue

        yolo_rows: list[tuple[int, float, float, float, float]] = []
        for ann in img_annots:
            md_cls = cat_map.get(ann.get("category_id"))
            if md_cls is None:
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            xc, yc, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
            if w <= 0 or h <= 0:
                continue
            yolo_rows.append((md_cls, xc, yc, w, h))

        if not yolo_rows:
            skipped_no_bbox += 1
            continue

        label_name = local_img.stem + ".txt"
        label_path = labels_dir / label_name
        write_yolo_label_file(label_path, yolo_rows)
        records_out.append({"image_path": local_img, "label_path": label_path})

    LOGGER.info(
        "WCS labels written: %d images  "
        "(%d skipped — not downloaded, %d skipped — no valid bbox)",
        len(records_out), skipped_no_download, skipped_no_bbox,
    )
    return records_out


# ---------------------------------------------------------------------------
# High-level entry point used by dataset_builder
# ---------------------------------------------------------------------------

def download_and_convert_wcs(
    output_images_dir: Path,
    output_labels_dir: Path,
    max_animal: int = 2500,
    max_vehicle: int = 500,
    seed: int = 42,
    num_workers: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_annotation_download: bool = False,
) -> list[dict]:
    """Full WCS pipeline: download → parse → sample → download images → labels.

    Returns list of {\"image_path\", \"label_path\"} dicts.
    """
    ann_path = download_wcs_annotations(
        cache_dir=cache_dir, force=force_annotation_download
    )

    with open(ann_path) as f:
        raw = json.load(f)
    cat_map = build_wcs_category_map(raw["categories"])

    image_records, annotations = sample_wcs_images(
        ann_path, max_animal=max_animal, max_vehicle=max_vehicle, seed=seed
    )

    id_to_path = download_wcs_images(
        image_records, output_images_dir, num_workers=num_workers
    )

    return build_wcs_label_files(
        image_records, annotations, id_to_path, output_labels_dir, cat_map
    )
