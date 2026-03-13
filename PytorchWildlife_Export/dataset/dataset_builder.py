"""
dataset_builder.py
------------------
Orchestrates the full dataset-building pipeline:

  1. Download + convert WCS Camera Traps (animals + vehicles).
  2. Download + convert COCO 2017 filtered subset (people + vehicles).
  3. Merge records, shuffle, split 80 / 10 / 10 into train / val / test.
  4. Lay out the YOLO directory structure (symlinks preferred, copies as fallback).
  5. Write megadetector_ft.yaml.

CLI usage
---------
    python -m PytorchWildlife_Export.dataset.dataset_builder \\
        --output-dir data/md_ft \\
        --wcs-max-animal 2500 \\
        --wcs-max-vehicle 500 \\
        --coco-max-person 1500 \\
        --coco-max-vehicle 500

Download budget (defaults)
--------------------------
    WCS  : ~3.5–5.0 GB  (2 500 animal + 500 vehicle images @ ~1.5 MB each)
    COCO : ~0.3–0.5 GB  (1 500 person + 500 vehicle images @ ~150 KB each)
    Total: ~4–5.5 GB — well within the 10 GB budget.

    WARNING: raising --wcs-max-animal above 3 500 risks approaching 10 GB.
    The script prints estimated image counts before downloading so you can
    abort (Ctrl-C) if the numbers look too high.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import shutil
from pathlib import Path

from .annotation_converter import MD_CLASS_NAMES, generate_dataset_yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/md_ft")
DEFAULT_SPLIT = (0.80, 0.10, 0.10)  # train / val / test


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _split_records(
    records: list[dict],
    split: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split records into train / val / test by ratio."""
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * split[0])
    n_val   = int(n * split[1])
    return (
        shuffled[:n_train],
        shuffled[n_train : n_train + n_val],
        shuffled[n_train + n_val :],
    )


# ---------------------------------------------------------------------------
# YOLO layout installer
# ---------------------------------------------------------------------------

def _install_yolo_layout(
    records: list[dict],
    split_name: str,
    output_dir: Path,
    use_symlinks: bool = True,
) -> None:
    """Place images + labels into the YOLO directory layout.

    output_dir/
        images/{split_name}/   ← symlink or copy of each image
        labels/{split_name}/   ← .txt label file (hard-linked or copied)
    """
    img_dir   = output_dir / "images" / split_name
    label_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        src_img   = Path(rec["image_path"])
        src_label = Path(rec["label_path"])
        dst_img   = img_dir   / src_img.name
        dst_label = label_dir / src_label.name

        # Image: prefer symlink (saves disk), fall back to copy
        if not dst_img.exists():
            if use_symlinks:
                try:
                    dst_img.symlink_to(src_img.resolve())
                except OSError:
                    shutil.copy2(src_img, dst_img)
            else:
                shutil.copy2(src_img, dst_img)

        # Label: always copy (small .txt files, no point symlinking)
        if not dst_label.exists():
            shutil.copy2(src_label, dst_label)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _print_summary(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    output_dir: Path,
) -> None:
    """Print per-split and per-class instance counts."""

    def count_classes(records: list[dict]) -> dict[int, int]:
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for rec in records:
            label_path = Path(rec["label_path"])
            if not label_path.exists():
                continue
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    cls = int(parts[0])
                    counts[cls] = counts.get(cls, 0) + 1
        return counts

    header = f"{'Split':<8}  {'Images':>7}  {'Animal':>8}  {'Person':>8}  {'Vehicle':>8}"
    print("\n" + "=" * 50)
    print("  Dataset build complete")
    print("=" * 50)
    print(header)
    for name, split_records in [("train", train), ("val", val), ("test", test)]:
        counts = count_classes(split_records)
        print(
            f"  {name:<6}  {len(split_records):>7}  "
            f"{counts.get(0,0):>8}  {counts.get(1,0):>8}  {counts.get(2,0):>8}"
        )
    total = len(train) + len(val) + len(test)
    print(f"  {'total':<6}  {total:>7}")
    print("=" * 50)
    print(f"  Config : {output_dir / 'megadetector_ft.yaml'}")
    print("=" * 50 + "\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def build_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    wcs_max_animal: int = 2500,
    wcs_max_vehicle: int = 500,
    coco_max_person: int = 1500,
    coco_max_vehicle: int = 500,
    split: tuple[float, float, float] = DEFAULT_SPLIT,
    seed: int = 42,
    num_download_workers: int = 8,
    skip_wcs: bool = False,
    skip_coco: bool = False,
    wcs_cache_dir: Path | None = None,
    use_symlinks: bool = True,
) -> Path:
    """Build the MegaDetector fine-tuning dataset.

    Downloads WCS and/or COCO subsets, merges them, splits into
    train / val / test, lays out the YOLO directory structure, and writes
    megadetector_ft.yaml.

    Parameters
    ----------
    output_dir:
        Root directory for the assembled dataset.
    wcs_max_animal:
        Maximum number of WCS images selected because they contain animals.
        Each image may contain multiple animal instances.  ~1.5 MB per image.
    wcs_max_vehicle:
        Maximum number of WCS images selected because they contain vehicles.
    coco_max_person:
        Maximum number of COCO images selected because they contain people.
        Each image is ~50–200 KB.
    coco_max_vehicle:
        Maximum number of COCO images selected because they contain vehicles.
    split:
        (train, val, test) fractions — must sum to 1.0.
    seed:
        Random seed for reproducible sampling and splitting.
    num_download_workers:
        Parallel HTTP workers for WCS image downloads.
    skip_wcs / skip_coco:
        Skip the respective source (useful for incremental builds or testing).
    wcs_cache_dir:
        Override the default WCS annotation cache directory.
    use_symlinks:
        Prefer symlinks over copies when installing the YOLO layout.

    Returns
    -------
    Path to the generated megadetector_ft.yaml.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    assert abs(sum(split) - 1.0) < 1e-6, "Split fractions must sum to 1.0"

    # Staging directories (raw downloads land here before split)
    raw_images_dir = output_dir / "_raw" / "images"
    raw_labels_dir = output_dir / "_raw" / "labels"

    all_records: list[dict] = []

    # ------------------------------------------------------------------
    # Source 1: WCS Camera Traps
    # ------------------------------------------------------------------
    if not skip_wcs:
        from .wcs_downloader import download_and_convert_wcs, DEFAULT_CACHE_DIR

        wcs_est = wcs_max_animal + wcs_max_vehicle
        LOGGER.info(
            "WCS: requesting up to %d animal images + %d vehicle images "
            "(~%.1f GB estimated).",
            wcs_max_animal, wcs_max_vehicle,
            wcs_est * 1.5 / 1024,  # rough 1.5 MB/image → GB
        )

        wcs_records = download_and_convert_wcs(
            output_images_dir=raw_images_dir / "wcs",
            output_labels_dir=raw_labels_dir / "wcs",
            max_animal=wcs_max_animal,
            max_vehicle=wcs_max_vehicle,
            seed=seed,
            num_workers=num_download_workers,
            cache_dir=wcs_cache_dir or DEFAULT_CACHE_DIR,
        )
        LOGGER.info("WCS: %d usable records.", len(wcs_records))
        all_records.extend(wcs_records)
    else:
        LOGGER.info("Skipping WCS download (--skip-wcs).")

    # ------------------------------------------------------------------
    # Source 2: COCO 2017 (filtered)
    # ------------------------------------------------------------------
    if not skip_coco:
        from .coco_downloader import download_and_convert_coco

        coco_est = coco_max_person + coco_max_vehicle
        LOGGER.info(
            "COCO: requesting up to %d person images + %d vehicle images "
            "(~%.0f MB estimated).",
            coco_max_person, coco_max_vehicle,
            coco_est * 0.15,  # rough 150 KB/image → MB
        )

        coco_records = download_and_convert_coco(
            output_images_dir=raw_images_dir / "coco",
            output_labels_dir=raw_labels_dir / "coco",
            max_person=coco_max_person,
            max_vehicle=coco_max_vehicle,
            seed=seed,
        )
        LOGGER.info("COCO: %d usable records.", len(coco_records))
        all_records.extend(coco_records)
    else:
        LOGGER.info("Skipping COCO download (--skip-coco).")

    if not all_records:
        raise RuntimeError(
            "No records collected — both WCS and COCO were skipped or produced "
            "no output.  Check logs for errors."
        )

    # ------------------------------------------------------------------
    # Split + install YOLO layout
    # ------------------------------------------------------------------
    LOGGER.info("Total records before split: %d", len(all_records))
    train_r, val_r, test_r = _split_records(all_records, split=split, seed=seed)
    LOGGER.info(
        "Split: train=%d  val=%d  test=%d", len(train_r), len(val_r), len(test_r)
    )

    for name, records in [("train", train_r), ("val", val_r), ("test", test_r)]:
        LOGGER.info("Installing %s split …", name)
        _install_yolo_layout(records, name, output_dir, use_symlinks=use_symlinks)

    # ------------------------------------------------------------------
    # Write dataset YAML
    # ------------------------------------------------------------------
    yaml_path = generate_dataset_yaml(output_dir, class_names=MD_CLASS_NAMES)

    # Print human-readable summary
    _print_summary(train_r, val_r, test_r, output_dir)

    return yaml_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MegaDetector fine-tuning dataset from WCS Camera Traps "
            "(animals + vehicles) and COCO 2017 (people + vehicles).\n\n"
            "BUDGET WARNING: default settings download ~4–5 GB.  "
            "Raising --wcs-max-animal above 3500 may approach the 10 GB limit."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help="Root directory for the assembled dataset. (default: %(default)s)",
    )
    parser.add_argument(
        "--wcs-max-animal", type=int, default=2500, metavar="N",
        help="Max WCS images selected for animal class. ~1.5 MB each. (default: %(default)s)",
    )
    parser.add_argument(
        "--wcs-max-vehicle", type=int, default=500, metavar="N",
        help="Max WCS images selected for vehicle class. (default: %(default)s)",
    )
    parser.add_argument(
        "--coco-max-person", type=int, default=1500, metavar="N",
        help="Max COCO images for person class. ~150 KB each. (default: %(default)s)",
    )
    parser.add_argument(
        "--coco-max-vehicle", type=int, default=500, metavar="N",
        help="Max COCO images for vehicle class. (default: %(default)s)",
    )
    parser.add_argument(
        "--split", type=float, nargs=3, default=list(DEFAULT_SPLIT),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train / val / test fractions. Must sum to 1. (default: 0.80 0.10 0.10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, metavar="N",
        help="Parallel HTTP workers for WCS image downloads. (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-wcs", action="store_true",
        help="Skip WCS download (use with an already-downloaded WCS subset).",
    )
    parser.add_argument(
        "--skip-coco", action="store_true",
        help="Skip COCO download (requires fiftyone not installed).",
    )
    parser.add_argument(
        "--no-symlinks", action="store_true",
        help="Copy files instead of symlinking them into the YOLO layout.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    split = tuple(args.split)
    if abs(sum(split) - 1.0) > 1e-6:
        raise SystemExit(
            f"ERROR: --split values {args.split} do not sum to 1.0 "
            f"(sum={sum(args.split):.4f})."
        )

    yaml_path = build_dataset(
        output_dir=args.output_dir,
        wcs_max_animal=args.wcs_max_animal,
        wcs_max_vehicle=args.wcs_max_vehicle,
        coco_max_person=args.coco_max_person,
        coco_max_vehicle=args.coco_max_vehicle,
        split=split,
        seed=args.seed,
        num_download_workers=args.workers,
        skip_wcs=args.skip_wcs,
        skip_coco=args.skip_coco,
        use_symlinks=not args.no_symlinks,
    )
    print(f"Dataset ready.  Config: {yaml_path}")
