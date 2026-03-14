"""
base_lila_downloader.py
-----------------------
Shared utilities for all LILA camera trap downloaders.

Provides:
  - download_annotation_file()       — cached download of JSON / ZIP-of-JSON
  - build_generic_category_map()     — maps categories to MD class ints
  - download_images_parallel()       — parallel image download via ThreadPoolExecutor
  - build_yolo_records()             — writes YOLO .txt files, returns record list
  - sample_empty_images()            — samples images with no annotations
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
from typing import Callable

from .annotation_converter import coco_bbox_to_yolo, wcs_category_to_md_class, write_yolo_label_file

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation download
# ---------------------------------------------------------------------------

def download_annotation_file(
    url: str,
    cache_path: Path,
    description: str,
) -> Path:
    """Download an annotation file (plain JSON or .zip containing one JSON).

    The result is cached at *cache_path*.  If *cache_path* already exists the
    download is skipped.  Supports .zip files that contain one or more .json
    files (they are merged if multiple).

    Returns the path to the local JSON file.
    """
    if cache_path.exists():
        LOGGER.info("%s annotations already cached at: %s", description, cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s annotation from:\n  %s", description, url)

    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            pct = min(100.0, block_num * block_size / total_size * 100)
            prev = (block_num - 1) * block_size / total_size * 100
            # print every ~10%
            if int(pct / 10) > int(prev / 10):
                LOGGER.info("  %.1f%%", pct)

    try:
        urllib.request.urlretrieve(url, tmp_path, _report)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to download {description} annotations from {url}.\n"
            f"Error: {exc}\n"
            f"Check https://lila.science for the current URL."
        ) from exc

    # If it's a zip, extract the JSON(s) and merge if needed
    if url.endswith(".zip") or zipfile.is_zipfile(tmp_path):
        _extract_annotation_zip(tmp_path, cache_path, description)
        tmp_path.unlink(missing_ok=True)
    else:
        tmp_path.rename(cache_path)

    LOGGER.info(
        "%s annotation cached: %s  (%.1f MB)",
        description,
        cache_path,
        cache_path.stat().st_size / 1e6,
    )
    return cache_path


def _extract_annotation_zip(zip_path: Path, dest_json_path: Path, description: str) -> None:
    """Extract JSON from a zip archive; merge multiple JSON files if present."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_names = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_names:
            raise RuntimeError(
                f"No .json files found in {description} zip archive."
            )

        if len(json_names) == 1:
            with zf.open(json_names[0]) as src, open(dest_json_path, "wb") as dst:
                dst.write(src.read())
            return

        # Multiple JSON files — merge them (e.g. Serengeti S1–S11)
        LOGGER.info(
            "Found %d JSON files in %s zip — merging …", len(json_names), description
        )
        merged: dict = {}
        for jname in sorted(json_names):
            LOGGER.info("  Merging %s", jname)
            with zf.open(jname) as f:
                data = json.load(f)
            if not merged:
                merged = {
                    "images": [],
                    "annotations": [],
                    "categories": data.get("categories", []),
                    "info": data.get("info", {}),
                }
            merged["images"].extend(data.get("images", []))
            merged["annotations"].extend(data.get("annotations", []))

        with open(dest_json_path, "w") as f:
            json.dump(merged, f)


# ---------------------------------------------------------------------------
# Category map
# ---------------------------------------------------------------------------

def build_generic_category_map(
    categories: list[dict],
    name_to_md_class_fn: Callable[[str], int | None],
) -> dict[int, int | None]:
    """Map a COCO-style categories list to MD class ints.

    Uses *name_to_md_class_fn* to convert each category name to an MD class
    (0=animal, 1=person, 2=vehicle, None=skip).
    """
    return {cat["id"]: name_to_md_class_fn(cat["name"]) for cat in categories}


# ---------------------------------------------------------------------------
# Parallel image download
# ---------------------------------------------------------------------------

def _download_one(
    image_id: str | int,
    url: str,
    dest_path: Path,
    timeout: int,
) -> tuple[str | int, bool]:
    """Download a single image.  Returns (image_id, success)."""
    if dest_path.exists():
        return image_id, True
    tmp = dest_path.with_suffix(dest_path.suffix + ".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pytorch-wildlife-onnx/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        tmp.rename(dest_path)
        return image_id, True
    except Exception as exc:
        LOGGER.debug("Failed to download %s: %s", url, exc)
        if tmp.exists():
            tmp.unlink()
        return image_id, False


def download_images_parallel(
    records: list[dict],
    dest_dir: Path,
    url_fn: Callable[[dict], str],
    num_workers: int = 8,
    timeout: int = 30,
) -> dict[str | int, Path]:
    """Download images in parallel using ThreadPoolExecutor.

    *records* is a list of image dicts (must have an 'id' key).
    *url_fn* takes an image record dict and returns the URL to download.

    Returns {image_id: local_path} for successfully downloaded images only.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    total = len(records)
    ok: dict[str | int, Path] = {}
    failed = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {}
        for rec in records:
            img_id = rec["id"]
            url = url_fn(rec)
            local_name = rec.get("_local_name") or Path(url).name
            dest_path = dest_dir / local_name
            fut = pool.submit(_download_one, img_id, url, dest_path, timeout)
            futures[fut] = (img_id, dest_path)

        for i, fut in enumerate(as_completed(futures), 1):
            img_id, dest_path = futures[fut]
            _, success = fut.result()
            if success:
                ok[img_id] = dest_path
            else:
                failed += 1
            if i % 50 == 0 or i == total:
                LOGGER.info(
                    "  Images: %d/%d done  (%d ok, %d failed)",
                    i, total, len(ok), failed,
                )

    LOGGER.info(
        "Image download complete: %d downloaded, %d failed.", len(ok), failed
    )
    return ok


# ---------------------------------------------------------------------------
# YOLO record builder
# ---------------------------------------------------------------------------

def build_yolo_records(
    image_records: list[dict],
    annotations_by_image_id: dict[str | int, list[dict]],
    id_to_local_path: dict[str | int, Path],
    labels_dir: Path,
    cat_map: dict[int, int | None],
    img_meta: dict[str | int, dict],
    source_name: str,
) -> list[dict]:
    """Write YOLO .txt label files and return a list of records.

    Each returned record has:
        image_path  : Path
        label_path  : Path
        source      : str   (source_name)
        location_id : str | None
        empty       : False (annotated images only; empty images handled separately)

    *image_records* — list of image dicts (must have 'id').
    *annotations_by_image_id* — {image_id: [ann_dict, ...]} with 'category_id' and 'bbox'.
    *id_to_local_path* — {image_id: local_image_path} for downloaded images.
    *labels_dir* — directory to write .txt files into.
    *cat_map* — {category_id: md_class | None}.
    *img_meta* — {image_id: {'width': int, 'height': int, 'location': ...}}.
    *source_name* — short string like "wcs", "cct", "ss".
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    skipped_not_downloaded = 0
    skipped_no_valid_bbox = 0

    for img_rec in image_records:
        img_id = img_rec["id"]
        local_path = id_to_local_path.get(img_id)
        if local_path is None:
            skipped_not_downloaded += 1
            continue

        meta = img_meta.get(img_id, {})
        img_w = meta.get("width")
        img_h = meta.get("height")

        # Try to get dims from the image file if not in metadata
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

        anns = annotations_by_image_id.get(img_id, [])
        yolo_anns: list[tuple[int, float, float, float, float]] = []
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

        location = meta.get("location")
        location_id = f"{source_name}:{location}" if location is not None else None

        records.append(
            {
                "image_path": local_path,
                "label_path": label_path,
                "source": source_name,
                "location_id": location_id,
                "empty": False,
            }
        )

    LOGGER.info(
        "%s labels written: %d images  (%d skipped — not downloaded, %d skipped — no valid bbox)",
        source_name,
        len(records),
        skipped_not_downloaded,
        skipped_no_valid_bbox,
    )
    return records


# ---------------------------------------------------------------------------
# Empty image sampling
# ---------------------------------------------------------------------------

def sample_empty_images(
    all_image_data: list[dict],
    annotated_ids: set,
    max_empty: int,
    seed: int,
) -> list[dict]:
    """Sample images that have no annotations.

    Returns a list of image dicts (with 'id', 'file_name', and any other fields).
    """
    unannotated = [img for img in all_image_data if img["id"] not in annotated_ids]
    rng = random.Random(seed)
    rng.shuffle(unannotated)
    return unannotated[:max_empty]
