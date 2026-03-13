"""
cct_downloader.py
-----------------
Downloads a budget-bounded subset of the Caltech Camera Traps (CCT20) dataset
from the LILA mirrors and converts annotations to YOLO format.

Intended use: OOD (out-of-distribution) validation against the QAT fine-tuned
MDV6-yolov10 model.  CCT20 covers SW USA wildlife (coyote, deer, rabbit,
squirrel, skunk, …) — completely different geography and species from the
WCS training data (African/global megafauna), making it a strong test of
genuine generalisation rather than in-distribution overfitting.

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
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import yaml

from .annotation_converter import (
    coco_bbox_to_yolo,
    wcs_category_to_md_class,
    write_yolo_label_file,
)

LOGGER = logging.getLogger(__name__)

CCT_ANNOTATION_URL = (
    "https://storage.googleapis.com/public-datasets-lila/"
    "caltechcameratraps/labels/caltech_bboxes_20200316.json"
)
CCT_IMAGE_BASE = (
    "https://lilawildlife.blob.core.windows.net/lila-wildlife/"
    "caltech-unzipped/cct_images"
)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorch_wildlife_export" / "cct"
CCT_ANNOTATION_FILENAME = "caltech_bboxes_20200316.json"


class CCTImageRecord(NamedTuple):
    image_id: str       # e.g. "59f79201-23d2-11e8-a6a3-ec086b02610b"
    file_name: str      # relative to annotation JSON (often same as image_id basename)
    width: int
    height: int


# ---------------------------------------------------------------------------
# Step 1: Download annotation JSON
# ---------------------------------------------------------------------------

def download_cct_annotations(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force: bool = False,
) -> Path:
    """Download the CCT20 bbox annotation JSON to *cache_dir*.

    The file is ~35 MB and is served directly (no ZIP).
    Returns the path to the local JSON file.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    json_path = cache_dir / CCT_ANNOTATION_FILENAME

    if json_path.exists() and not force:
        LOGGER.info("CCT annotations already cached at: %s", json_path)
        return json_path

    LOGGER.info(
        "Downloading CCT20 annotation JSON from:\n  %s", CCT_ANNOTATION_URL
    )
    LOGGER.info("File is ~35 MB — this may take a minute on a slow connection.")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0 and block_num % 500 == 0:
            pct = min(100.0, downloaded / total_size * 100)
            LOGGER.info("  %.1f%%", pct)

    tmp_path = json_path.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(CCT_ANNOTATION_URL, tmp_path, reporthook=_report)
        tmp_path.rename(json_path)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f"Failed to download CCT annotations from {CCT_ANNOTATION_URL}.\n"
            f"Error: {exc}\n"
            "Check https://lila.science/datasets/caltech-camera-traps for current URLs."
        ) from exc

    LOGGER.info(
        "CCT annotation JSON cached: %s  (%.1f MB)",
        json_path, json_path.stat().st_size / 1024 / 1024,
    )
    return json_path


# ---------------------------------------------------------------------------
# Step 2: Parse + sample
# ---------------------------------------------------------------------------

def _build_cct_category_map(categories: list[dict]) -> dict[int, int | None]:
    """Map CCT category_id → MD class (0=animal, 1=person, None=skip).

    CCT uses fine-grained species names.  We reuse `wcs_category_to_md_class`
    which already handles the animal/person/vehicle/skip classification by
    keyword matching.  CCT has no vehicle category — those are just skipped.
    """
    return {
        cat["id"]: wcs_category_to_md_class(cat["name"])
        for cat in categories
    }


def sample_cct_images(
    annotation_path: Path,
    max_animal: int = 500,
    seed: int = 42,
) -> tuple[list[CCTImageRecord], list[dict], dict[int, int | None]]:
    """Parse CCT annotations and select a budget-bounded subset.

    Only animal-primary images are sampled (CCT is almost exclusively wildlife).

    Returns
    -------
    (image_records, filtered_annotations, category_map)
    """
    LOGGER.info("Parsing CCT annotation JSON: %s", annotation_path)
    with open(annotation_path) as f:
        data = json.load(f)

    cat_map = _build_cct_category_map(data["categories"])

    n_animal  = sum(1 for v in cat_map.values() if v == 0)
    n_person  = sum(1 for v in cat_map.values() if v == 1)
    n_skip    = sum(1 for v in cat_map.values() if v is None)
    LOGGER.info(
        "CCT categories: %d total → animal=%d  person=%d  skip=%d",
        len(cat_map), n_animal, n_person, n_skip,
    )

    image_by_id: dict[str, dict] = {img["id"]: img for img in data["images"]}

    # Index annotations per image; keep animal + person (skip empty/unknown)
    annots_by_image: dict[str, list[dict]] = {}
    for ann in data["annotations"]:
        md_cls = cat_map.get(ann.get("category_id"))
        if md_cls is not None:  # 0 = animal, 1 = person
            annots_by_image.setdefault(ann["image_id"], []).append(ann)

    # Focus on animal-primary images (the OOD goal)
    animal_image_ids = [
        img_id for img_id, anns in annots_by_image.items()
        if any(cat_map.get(a.get("category_id")) == 0 for a in anns)
    ]
    LOGGER.info(
        "Images with ≥1 animal annotation: %d  (requesting %d)",
        len(animal_image_ids), max_animal,
    )

    rng = random.Random(seed)
    rng.shuffle(animal_image_ids)
    selected_ids = set(animal_image_ids[:max_animal])

    image_records: list[CCTImageRecord] = []
    for img_id in selected_ids:
        img = image_by_id.get(img_id)
        if img is None:
            continue
        image_records.append(
            CCTImageRecord(
                image_id=img["id"],
                file_name=img.get("file_name", img["id"] + ".jpg"),
                width=img.get("width", 0),
                height=img.get("height", 0),
            )
        )

    filtered_annotations: list[dict] = []
    for img_id in selected_ids:
        filtered_annotations.extend(annots_by_image.get(img_id, []))

    LOGGER.info(
        "Selected %d images with %d annotations.",
        len(image_records), len(filtered_annotations),
    )
    return image_records, filtered_annotations, cat_map


# ---------------------------------------------------------------------------
# Step 3: Download images
# ---------------------------------------------------------------------------

def _download_one_cct(
    record: CCTImageRecord,
    dest_path: Path,
    timeout: int,
) -> tuple[CCTImageRecord, bool]:
    """Download a single CCT image.  Returns (record, success)."""
    if dest_path.exists():
        return record, True
    # CCT image URL: {CCT_IMAGE_BASE}/{image_id}.jpg
    # The image_id is the bare UUID; some file_names include subdirectories but
    # the Azure flat mirror uses just the UUID basename.
    basename = Path(record.file_name).name
    url = f"{CCT_IMAGE_BASE}/{basename}"
    tmp = dest_path.with_suffix(".tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "pytorch-wildlife-onnx/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as f:
            f.write(resp.read())
        tmp.rename(dest_path)
        return record, True
    except Exception as exc:
        LOGGER.debug("Failed to download %s: %s", url, exc)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return record, False


def download_cct_images(
    image_records: list[CCTImageRecord],
    images_dir: Path,
    num_workers: int = 8,
    timeout: int = 30,
) -> dict[str, Path]:
    """Download CCT images with parallel workers.

    Images are stored flat under *images_dir* as ``{image_id}.jpg``.
    Returns {image_id: local_path} for successfully downloaded images only.
    """
    images_dir.mkdir(parents=True, exist_ok=True)

    id_to_path: dict[str, Path] = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        for rec in image_records:
            # Use just the basename (UUID.jpg) to keep the directory flat
            basename = Path(rec.file_name).name
            dest = images_dir / basename
            id_to_path[rec.image_id] = dest
            fut = pool.submit(_download_one_cct, rec, dest, timeout)
            futures[fut] = rec

        ok = fail = 0
        for i, fut in enumerate(as_completed(futures), 1):
            rec, success = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
                id_to_path.pop(rec.image_id, None)
            if i % 50 == 0 or i == len(futures):
                LOGGER.info(
                    "  Images: %d/%d done  (%d ok, %d failed)",
                    i, len(futures), ok, fail,
                )

    LOGGER.info("CCT image download complete: %d downloaded, %d failed.", ok, fail)
    return id_to_path


# ---------------------------------------------------------------------------
# Step 4: Write YOLO labels
# ---------------------------------------------------------------------------

def build_cct_label_files(
    image_records: list[CCTImageRecord],
    annotations: list[dict],
    id_to_local_path: dict[str, Path],
    labels_dir: Path,
    cat_map: dict[int, int | None],
) -> list[dict]:
    """Write YOLO .txt label files for downloaded CCT images.

    Returns a list of {"image_path": Path, "label_path": Path} dicts.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)

    annots_by_image: dict[str, list[dict]] = {}
    for ann in annotations:
        annots_by_image.setdefault(ann["image_id"], []).append(ann)

    img_meta: dict[str, CCTImageRecord] = {r.image_id: r for r in image_records}

    records_out: list[dict] = []
    skipped_no_download = 0
    skipped_no_bbox = 0

    for img_id, img_annots in annots_by_image.items():
        local_img = id_to_local_path.get(img_id)
        if local_img is None:
            skipped_no_download += 1
            continue

        rec = img_meta.get(img_id)
        img_w = rec.width  if rec else 0
        img_h = rec.height if rec else 0

        # Fall back to cv2 if dimensions not in annotation
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

            # Prefer absolute bbox; fall back to bbox_relative (already normalised)
            bbox = ann.get("bbox")
            if bbox and len(bbox) >= 4 and img_w > 0 and img_h > 0:
                xc, yc, w, h = coco_bbox_to_yolo(bbox, img_w, img_h)
            else:
                bbox_rel = ann.get("bbox_relative")
                if not bbox_rel or len(bbox_rel) < 4:
                    continue
                # bbox_relative = [x_min, y_min, w, h] normalised → convert to center
                x, y, bw, bh = bbox_rel
                xc = x + bw / 2
                yc = y + bh / 2
                w, h = bw, bh

            if w <= 0 or h <= 0:
                continue
            yolo_rows.append((md_cls, xc, yc, w, h))

        if not yolo_rows:
            skipped_no_bbox += 1
            continue

        label_path = labels_dir / (local_img.stem + ".txt")
        write_yolo_label_file(label_path, yolo_rows)
        records_out.append({"image_path": local_img, "label_path": label_path})

    LOGGER.info(
        "CCT labels written: %d images  "
        "(%d skipped — not downloaded, %d skipped — no valid bbox)",
        len(records_out), skipped_no_download, skipped_no_bbox,
    )
    return records_out


# ---------------------------------------------------------------------------
# Step 5: Write dataset YAML
# ---------------------------------------------------------------------------

def write_cct_dataset_yaml(output_dir: Path) -> Path:
    """Write a minimal Ultralytics-compatible dataset YAML for CCT OOD eval.

    All images are placed under images/val — this dataset is eval-only.
    """
    config = {
        "path": str(output_dir.resolve()),
        "val":  "images/val",
        "nc":   3,
        "names": {0: "animal", 1: "person", 2: "vehicle"},
    }
    yaml_path = output_dir / "cct_ood.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    LOGGER.info("CCT OOD dataset YAML written: %s", yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def download_and_convert_cct(
    output_dir: Path,
    max_animal: int = 500,
    seed: int = 42,
    num_workers: int = 8,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_annotation_download: bool = False,
) -> tuple[list[dict], Path]:
    """Full CCT OOD pipeline: download → parse → sample → download images → labels → YAML.

    Parameters
    ----------
    output_dir:
        Root directory for the assembled dataset.
        Will be created if it does not exist.
        Layout:
            output_dir/images/val/   ← downloaded images
            output_dir/labels/val/   ← YOLO .txt label files
            output_dir/cct_ood.yaml  ← Ultralytics dataset config
    max_animal:
        Maximum number of animal-primary images to download.
    seed:
        Random seed for reproducible sampling.
    num_workers:
        Parallel HTTP workers for image downloads.
    cache_dir:
        Directory for caching the annotation JSON.
    force_annotation_download:
        Re-download the annotation JSON even if cached.

    Returns
    -------
    (records, yaml_path) where records is a list of
    {"image_path": Path, "label_path": Path} dicts.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ann_path = download_cct_annotations(
        cache_dir=cache_dir, force=force_annotation_download
    )

    image_records, annotations, cat_map = sample_cct_images(
        ann_path, max_animal=max_animal, seed=seed
    )

    images_dir = output_dir / "images" / "val"
    labels_dir = output_dir / "labels" / "val"

    id_to_path = download_cct_images(
        image_records, images_dir, num_workers=num_workers
    )

    records = build_cct_label_files(
        image_records, annotations, id_to_path, labels_dir, cat_map
    )

    yaml_path = write_cct_dataset_yaml(output_dir)

    LOGGER.info(
        "CCT OOD dataset ready: %d images.  YAML: %s", len(records), yaml_path
    )
    return records, yaml_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description=(
            "Download a sample of the Caltech Camera Traps (CCT20) dataset "
            "for OOD evaluation of MDV6-yolov10 models.\n\n"
            "Downloads ~500 animal images from SW USA trail cameras — completely "
            "different geography and species from the WCS training data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/cct_ood"),
        metavar="DIR",
        help="Root directory for the assembled dataset. (default: %(default)s)",
    )
    parser.add_argument(
        "--max-animal", type=int, default=500, metavar="N",
        help="Max animal-primary images to download. (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="N",
        help="Parallel HTTP workers. (default: %(default)s)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download annotation JSON even if cached.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


if __name__ == "__main__":
    import logging
    import sys
    from pathlib import Path as _Path

    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Add project root to sys.path so sibling imports work
    _proj_root = str(_Path(__file__).resolve().parents[2])
    if _proj_root not in sys.path:
        sys.path.insert(0, _proj_root)

    records, yaml_path = download_and_convert_cct(
        output_dir=args.output_dir,
        max_animal=args.max_animal,
        seed=args.seed,
        num_workers=args.workers,
        force_annotation_download=args.force,
    )
    print(f"\nCCT OOD dataset ready: {len(records)} images")
    print(f"YAML:  {yaml_path}")
    print(f"\nEval usage:")
    print(f"  make eval-ood MODEL=<your_model>.engine")
    print(f"  # or:")
    print(
        f"  python -m PytorchWildlife_Export.dataset.eval <model> "
        f"--dataset {yaml_path} --split val"
    )
