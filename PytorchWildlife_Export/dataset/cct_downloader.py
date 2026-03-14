"""
cct_downloader.py
-----------------
Downloads a budget-bounded subset of the Caltech Camera Traps (CCT20) dataset
from the LILA mirrors and converts annotations to YOLO format.

Data source
    LILA Caltech Camera Traps
    https://lila.science/datasets/caltech-camera-traps

    Annotation JSON (direct HTTP, no auth, ~35 MB):
        https://storage.googleapis.com/public-datasets-lila/
            caltechcameratraps/labels/caltech_bboxes_20200316.json

    Images (Azure LILA mirror, no auth):
        https://lilawildlife.blob.core.windows.net/lila-wildlife/
            caltech-unzipped/cct_images/{image_id}.jpg

Annotation format
    COCO Camera Traps JSON with fine-grained species categories.
    Category name → MegaDetector class:
        "empty" / "unknown" / setup terms  → skip (no label)
        "person" / "people" / "human" …    → MD class 1 (person)
        all other wildlife species          → MD class 0 (animal)
    NOTE: CCT20 contains virtually no vehicles; no vehicle sampling is done.

Budget
    Annotation JSON: ~35 MB (single file, no ZIP extraction needed).
    Images: ~300–700 KB each (trail-cam JPEGs, typically 1–2 MP).
    Default 500 images ≈ 200–350 MB download — well within budget.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import NamedTuple

from .annotation_converter import coco_bbox_to_yolo, wcs_category_to_md_class, write_yolo_label_file
from .base_lila_downloader import (
    download_annotation_file,
    download_images_parallel,
    sample_empty_images,
)

LOGGER = logging.getLogger(__name__)

CCT_ANNOTATION_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_bboxes_20200316.json"
)
CCT_IMAGE_BASE = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images"
)
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorch_wildlife_export" / "cct"
CCT_ANNOTATION_FILENAME = "caltech_bboxes_20200316.json"


class CCTImageRecord(NamedTuple):
    image_id: str
    file_name: str
    width: int
    height: int


# ---------------------------------------------------------------------------
# Annotation download
# ---------------------------------------------------------------------------

def download_cct_annotations(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Path:
    """Download the CCT20 bbox annotation JSON to *cache_dir*."""
    cache_dir = Path(cache_dir)
    json_path = cache_dir / CCT_ANNOTATION_FILENAME
    if force and json_path.exists():
        json_path.unlink()
    try:
        return download_annotation_file(
            url=CCT_ANNOTATION_URL,
            cache_path=json_path,
            description="CCT",
        )
    except RuntimeError as exc:
        LOGGER.error(str(exc))
        raise


# ---------------------------------------------------------------------------
# Category map
# ---------------------------------------------------------------------------

def _build_cct_category_map(categories: list[dict]) -> dict[int, int | None]:
    """Map CCT category_id → MD class (0=animal, 1=person, None=skip).

    CCT uses fine-grained species names.  We reuse `wcs_category_to_md_class`
    which handles the same vocabulary.
    """
    return {cat["id"]: wcs_category_to_md_class(cat["name"]) for cat in categories}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_cct_images(
    annotation_path: Path,
    max_animal: int = 500,
    seed: int = 42,
) -> tuple[list[CCTImageRecord], list[dict], dict[int, int | None]]:
    """Parse CCT annotations and select a budget-bounded subset.

    Only animal-primary images are sampled (CCT is almost exclusively animals).
    Returns (image_records, all_annotations, cat_map).
    """
    LOGGER.info("Parsing CCT annotation JSON: %s", annotation_path)
    with open(annotation_path) as f:
        data = json.load(f)

    cat_map = _build_cct_category_map(data["categories"])
    n_animal = sum(1 for v in cat_map.values() if v == 0)
    n_person = sum(1 for v in cat_map.values() if v == 1)
    n_skip = sum(1 for v in cat_map.values() if v is None)
    LOGGER.info(
        "CCT categories: %d total → animal=%d  person=%d  skip=%d",
        len(cat_map), n_animal, n_person, n_skip,
    )

    # Group annotations by image_id
    anns_by_img: dict[str, list[int]] = {}
    for ann in data["annotations"]:
        cls = cat_map.get(ann["category_id"])
        if cls is not None:
            anns_by_img.setdefault(ann["image_id"], []).append(cls)

    # Collect image ids with at least one animal annotation
    animal_img_ids = [
        img_id for img_id, classes in anns_by_img.items() if any(c == 0 for c in classes)
    ]
    LOGGER.info(
        "Images with ≥1 animal annotation: %d  (requesting %d)",
        len(animal_img_ids), max_animal,
    )

    rng = random.Random(seed)
    rng.shuffle(animal_img_ids)
    selected_ids = set(animal_img_ids[:max_animal])

    img_by_id = {img["id"]: img for img in data["images"]}
    records: list[CCTImageRecord] = []
    for img_id in selected_ids:
        img = img_by_id.get(img_id)
        if img is None:
            continue
        fname = img.get("file_name") or (img_id + ".jpg")
        records.append(
            CCTImageRecord(
                image_id=img["id"],
                file_name=fname,
                width=img.get("width", 0),
                height=img.get("height", 0),
            )
        )

    n_anns = sum(1 for ann in data["annotations"] if ann["image_id"] in selected_ids)
    LOGGER.info("Selected %d images with %d annotations.", len(records), n_anns)
    return records, data["annotations"], cat_map


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

def download_cct_images(
    image_records: list[CCTImageRecord],
    images_dir: Path,
    num_workers: int = 8,
    timeout: int = 30,
) -> dict[str, Path]:
    """Download CCT images with parallel workers.

    Images are stored flat under *images_dir* as ``cct_{image_id}.jpg``.
    Returns {image_id: local_path} for successfully downloaded images.
    """
    dl_records = []
    for rec in image_records:
        local_name = f"cct_{rec.image_id}.jpg"
        url = f"{CCT_IMAGE_BASE}/{rec.image_id}.jpg"
        dl_records.append({
            "id": rec.image_id,
            "file_name": rec.file_name,
            "_local_name": local_name,
            "_url": url,
        })

    def url_fn(r: dict) -> str:
        return r["_url"]

    return download_images_parallel(dl_records, images_dir, url_fn, num_workers, timeout)


# ---------------------------------------------------------------------------
# Label file builder
# ---------------------------------------------------------------------------

def build_cct_label_files(
    image_records: list[CCTImageRecord],
    annotations: list[dict],
    id_to_local_path: dict[str, Path],
    labels_dir: Path,
    cat_map: dict[int, int | None],
    img_location_map: dict[str, int | None] | None = None,
) -> list[dict]:
    """Write YOLO .txt label files for downloaded CCT images.

    Returns a list of records with keys:
        image_path, label_path, source="cct", location_id, empty=False
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    if img_location_map is None:
        img_location_map = {}

    anns_by_img: dict[str, list[dict]] = {}
    for ann in annotations:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

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

        location = img_location_map.get(img_id)
        location_id = f"cct:{location}" if location is not None else None

        records.append(
            {
                "image_path": local_path,
                "label_path": label_path,
                "source": "cct",
                "location_id": location_id,
                "empty": False,
            }
        )

    LOGGER.info(
        "CCT labels written: %d images  (%d skipped — not downloaded, %d skipped — no valid bbox)",
        len(records), skipped_not_downloaded, skipped_no_valid_bbox,
    )
    return records


# ---------------------------------------------------------------------------
# Empty image helpers
# ---------------------------------------------------------------------------

def _download_empty_cct(
    annotation_path: Path,
    annotated_ids: set,
    images_dir: Path,
    labels_dir: Path,
    max_empty: int,
    seed: int,
    num_workers: int,
    timeout: int,
    img_location_map: dict,
) -> list[dict]:
    """Download and build records for empty (no-annotation) CCT images."""
    with open(annotation_path) as f:
        data = json.load(f)

    empty_imgs = sample_empty_images(data["images"], annotated_ids, max_empty, seed)
    LOGGER.info("Sampling %d empty CCT images.", len(empty_imgs))

    dl_records = []
    for img in empty_imgs:
        img_id = img["id"]
        local_name = f"cct_{img_id}.jpg"
        url = f"{CCT_IMAGE_BASE}/{img_id}.jpg"
        dl_records.append({
            "id": img_id,
            "_local_name": local_name,
            "_url": url,
            "_location": img_location_map.get(img_id),
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
        label_path.touch()
        location = r.get("_location")
        location_id = f"cct:{location}" if location is not None else None
        records.append(
            {
                "image_path": local_path,
                "label_path": label_path,
                "source": "cct",
                "location_id": location_id,
                "empty": True,
            }
        )

    LOGGER.info("CCT empty records: %d", len(records))
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_and_convert_cct(
    output_dir: Path,
    max_animal: int = 500,
    max_empty: int = 0,
    seed: int = 42,
    num_workers: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_annotation_download: bool = False,
) -> list[dict]:
    """Full CCT pipeline: download → parse → sample → download images → labels.

    Returns list of records with keys:
        image_path, label_path, source="cct", location_id, empty

    The dataset_builder handles all train/val/test splitting based on location_id.
    """
    output_dir = Path(output_dir).resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_path = download_cct_annotations(cache_dir, force=force_annotation_download)

    with open(annotation_path) as f:
        data = json.load(f)

    # Build location map
    img_location_map: dict[str, int | None] = {}
    for img in data["images"]:
        img_location_map[img["id"]] = img.get("location")

    image_records, annotations, cat_map = sample_cct_images(
        annotation_path, max_animal, seed
    )

    id_to_local_path = download_cct_images(image_records, images_dir, num_workers)

    records = build_cct_label_files(
        image_records, annotations, id_to_local_path, labels_dir, cat_map, img_location_map
    )

    annotated_ids = {rec.image_id for rec in image_records}

    if max_empty > 0:
        empty_records = _download_empty_cct(
            annotation_path,
            annotated_ids,
            images_dir,
            labels_dir,
            max_empty,
            seed,
            num_workers,
            30,
            img_location_map,
        )
        records.extend(empty_records)

    LOGGER.info("CCT dataset ready: %d images.", len(records))
    return records


# ---------------------------------------------------------------------------
# CLI (backwards compat)
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(
        description=(
            "Download a sample of the Caltech Camera Traps (CCT20) dataset.\n\n"
            "Downloads CCT images and writes YOLO format labels.\n"
            "Use --output-dir to specify destination."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", default="data/cct_ood", type=Path,
                   help="Root directory for the assembled dataset. (default: %(default)s)")
    p.add_argument("--max-animal", default=500, type=int,
                   help="Max animal-primary images to download. (default: %(default)s)")
    p.add_argument("--max-empty", default=0, type=int,
                   help="Max empty images to download. (default: %(default)s)")
    p.add_argument("--seed", default=42, type=int,
                   help="Random seed for reproducible sampling. (default: %(default)s)")
    p.add_argument("--workers", default=8, type=int,
                   help="Parallel HTTP workers. (default: %(default)s)")
    p.add_argument("--force", action="store_true",
                   help="Re-download annotation JSON even if cached.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


if __name__ == "__main__":
    import logging as _logging
    args = _parse_args()
    _logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(_logging, args.log_level.upper(), _logging.INFO),
    )
    records = download_and_convert_cct(
        output_dir=args.output_dir,
        max_animal=args.max_animal,
        max_empty=args.max_empty,
        seed=args.seed,
        num_workers=args.workers,
        force_annotation_download=args.force,
    )
    print(f"\nCCT dataset ready: {len(records)} images")
    print(f"  make eval-ood MODEL=<your_model>.engine")
