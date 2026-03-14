"""
dataset_builder.py
------------------
Orchestrates the full dataset-building pipeline:

  1. Download + convert WCS Camera Traps (animals + vehicles).
  2. Download + convert CCT (Caltech Camera Traps).
  3. Download + convert Snapshot Serengeti.
  4. Download + convert Island Conservation Camera Traps.
  5. Download + convert COCO 2017 filtered subset (people + vehicles).
  7. Merge all records, assign locations to splits (80/10/10 by location).
  8. Lay out YOLO directory structure (symlinks preferred, copies as fallback).
  9. Write megadetector_ft.yaml.
  10. Write data_readme.md with full provenance.

CLI usage
---------
    python -m PytorchWildlife_Export.dataset.dataset_builder \\
        --output-dir data/md_ft \\
        --wcs-max-animal 800 \\
        --cct-max-animal 700 \\
        --coco-max-person 500 \\
        --coco-max-vehicle 300
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
from collections import defaultdict
from datetime import date
from pathlib import Path

from .annotation_converter import MD_CLASS_NAMES, generate_dataset_yaml

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("data/md_ft")
DEFAULT_SPLIT = (0.80, 0.10, 0.10)

SOURCE_DESCRIPTIONS = {
    "wcs": ("WCS Camera Traps", "https://lila.science/datasets/wcscameratraps"),
    "cct": ("Caltech Camera Traps", "https://lila.science/datasets/caltech-camera-traps"),
    "ss": ("Snapshot Serengeti", "https://lila.science/datasets/snapshot-serengeti"),
    "ict": ("Island Conservation", "https://lila.science/datasets/island-conservation-camera-traps"),
    "coco": ("COCO 2017 (filtered)", "https://cocodataset.org/"),
}


# ---------------------------------------------------------------------------
# Location-aware split
# ---------------------------------------------------------------------------

def _assign_locations(
    records: list[dict],
    split: tuple[float, float, float] = (0.80, 0.10, 0.10),
    seed: int = 42,
) -> dict[str, str]:
    """Assign each unique location_id to exactly one split.

    Returns {location_id: split_name} where split_name is one of
    "train", "val", "test".
    """
    loc_to_count: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("location_id"):
            loc_to_count[r["location_id"]] += 1

    locations = sorted(loc_to_count)
    rng = random.Random(seed)
    rng.shuffle(locations)

    n = len(locations)
    n_train = int(n * split[0])
    n_val = int(n * split[1])

    assignments: dict[str, str] = {}
    for i, loc in enumerate(locations):
        if i < n_train:
            assignments[loc] = "train"
        elif i < n_train + n_val:
            assignments[loc] = "val"
        else:
            assignments[loc] = "test"

    return assignments


def _save_location_assignments(path: Path, assignments: dict[str, str]) -> None:
    """Save location assignments to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "generated": str(date.today()),
        "note": (
            "location_id -> split assignment; load with --location-assignments "
            "to reuse when adding more images"
        ),
        "assignments": assignments,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    LOGGER.info("Location assignments saved: %s", path)


def _load_location_assignments(path: Path) -> dict[str, str]:
    """Load location assignments from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    assignments = data.get("assignments", data)  # support both formats
    LOGGER.info("Loaded %d location assignments from %s", len(assignments), path)
    return assignments


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def _split_records(
    records: list[dict],
    split: tuple[float, float, float] = (0.80, 0.10, 0.10),
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Shuffle and split records into train / val / test by ratio.

    Used for records that have no location_id (e.g. COCO).
    """
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    n = len(records)
    n_train = int(n * split[0])
    n_val = int(n * split[1])
    return records[:n_train], records[n_train:n_train + n_val], records[n_train + n_val:]


def _apply_location_split(
    records: list[dict],
    assignments: dict[str, str],
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Assign records to splits based on location assignments.

    Records with unknown location_id are assigned randomly.
    """
    train, val, test = [], [], []
    unknown: list[dict] = []

    for r in records:
        loc = r.get("location_id")
        if loc and loc in assignments:
            split_name = assignments[loc]
            if split_name == "train":
                train.append(r)
            elif split_name == "val":
                val.append(r)
            else:
                test.append(r)
        else:
            unknown.append(r)

    if unknown:
        LOGGER.warning(
            "%d records have no/unknown location_id — splitting randomly.", len(unknown)
        )
        rng = random.Random(seed)
        rng.shuffle(unknown)
        n = len(unknown)
        n_train = int(n * 0.80)
        n_val = int(n * 0.10)
        train.extend(unknown[:n_train])
        val.extend(unknown[n_train:n_train + n_val])
        test.extend(unknown[n_train + n_val:])

    return train, val, test


# ---------------------------------------------------------------------------
# YOLO directory layout
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
        labels/{split_name}/   ← symlink or copy of each label
    """
    img_dir = output_dir / "images" / split_name
    lbl_dir = output_dir / "labels" / split_name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for rec in records:
        img_src = Path(rec["image_path"])
        lbl_src = Path(rec["label_path"])

        img_dst = img_dir / img_src.name
        lbl_dst = lbl_dir / lbl_src.name

        for src, dst in [(img_src, img_dst), (lbl_src, lbl_dst)]:
            if dst.exists():
                continue
            if use_symlinks:
                try:
                    dst.symlink_to(src.resolve())
                    continue
                except OSError:
                    pass
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Summary / README
# ---------------------------------------------------------------------------

def _count_classes(records: list[dict]) -> dict[int, int]:
    """Count annotation instances by MD class across a list of records."""
    counts: dict[int, int] = defaultdict(int)
    for rec in records:
        lbl_path = Path(rec["label_path"])
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                cls_id = int(line.split()[0])
                counts[cls_id] += 1
            except (ValueError, IndexError):
                pass
    return dict(counts)


def _count_locations(records: list[dict]) -> int:
    return len({r["location_id"] for r in records if r.get("location_id")})


def _print_summary(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    output_dir: Path,
) -> None:
    """Print per-split and per-class instance counts."""
    splits = [("train", train), ("val", val), ("test", test)]
    all_records = train + val + test

    print("\n" + "=" * 60)
    print("  Dataset build complete")
    print("=" * 60)
    header = f"{'Split':<8} {'Images':>7} {'Animal':>7} {'Person':>7} {'Vehicle':>7} {'Empty':>7} {'Locs':>6}"
    print(header)
    print("-" * 60)

    totals = {"images": 0, "animal": 0, "person": 0, "vehicle": 0, "empty": 0, "locs": 0}
    for name, records in splits:
        counts = _count_classes(records)
        n_empty = sum(1 for r in records if r.get("empty"))
        n_locs = _count_locations(records)
        row = (
            f"{name:<8} {len(records):>7} {counts.get(0,0):>7} "
            f"{counts.get(1,0):>7} {counts.get(2,0):>7} {n_empty:>7} {n_locs:>6}"
        )
        print(row)
        totals["images"] += len(records)
        totals["animal"] += counts.get(0, 0)
        totals["person"] += counts.get(1, 0)
        totals["vehicle"] += counts.get(2, 0)
        totals["empty"] += n_empty
        totals["locs"] += n_locs

    print("-" * 60)
    print(
        f"{'total':<8} {totals['images']:>7} {totals['animal']:>7} "
        f"{totals['person']:>7} {totals['vehicle']:>7} {totals['empty']:>7} {totals['locs']:>6}"
    )
    print("=" * 60)
    print(f"  Config : {output_dir / 'megadetector_ft.yaml'}")
    print("=" * 60 + "\n")


def _write_readme(
    output_dir: Path,
    all_records: list[dict],
    train_r: list[dict],
    val_r: list[dict],
    test_r: list[dict],
    location_assignments: dict[str, str],
) -> Path:
    """Write data_readme.md with provenance, split summary, and location assignments."""
    today = str(date.today())

    # Per-source stats
    source_stats: dict[str, dict] = {}
    for rec in all_records:
        src = rec.get("source", "unknown")
        if src not in source_stats:
            source_stats[src] = {"total": 0, "train": 0, "val": 0, "test": 0, "locs": set()}
        source_stats[src]["total"] += 1
        loc = rec.get("location_id")
        if loc:
            source_stats[src]["locs"].add(loc)

    for rec in train_r:
        source_stats.get(rec.get("source", ""), {})
        src = rec.get("source", "unknown")
        if src in source_stats:
            source_stats[src]["train"] += 1
    for rec in val_r:
        src = rec.get("source", "unknown")
        if src in source_stats:
            source_stats[src]["val"] += 1
    for rec in test_r:
        src = rec.get("source", "unknown")
        if src in source_stats:
            source_stats[src]["test"] += 1

    # Val/test location assignments per source
    val_locs_by_source: dict[str, list[str]] = defaultdict(list)
    test_locs_by_source: dict[str, list[str]] = defaultdict(list)
    for loc_id, split_name in sorted(location_assignments.items()):
        src = loc_id.split(":")[0] if ":" in loc_id else "unknown"
        if split_name == "val":
            val_locs_by_source[src].append(loc_id)
        elif split_name == "test":
            test_locs_by_source[src].append(loc_id)

    lines: list[str] = []
    lines.append(f"# MegaDetector Fine-tuning Dataset")
    lines.append(f"")
    lines.append(f"Generated: {today}")
    lines.append(f"")
    lines.append(f"## Provenance")
    lines.append(f"")
    lines.append(f"| Source | Description | URL | N Images | N Locations | Split Method |")
    lines.append(f"|--------|-------------|-----|----------|-------------|--------------|")

    for src, stats in source_stats.items():
        desc, url = SOURCE_DESCRIPTIONS.get(src, (src, ""))
        n_locs = len(stats["locs"])
        split_method = "location-aware" if n_locs > 0 else "random"
        lines.append(
            f"| {src} | {desc} | [{url}]({url}) | {stats['total']} | {n_locs} | {split_method} |"
        )

    lines.append(f"")
    lines.append(f"## Split Summary")
    lines.append(f"")
    lines.append(f"| Split | N Images | N Animal | N Person | N Vehicle | N Empty | N Locations |")
    lines.append(f"|-------|----------|----------|----------|-----------|---------|-------------|")

    for name, records in [("train", train_r), ("val", val_r), ("test", test_r), ("total", all_records)]:
        counts = _count_classes(records)
        n_empty = sum(1 for r in records if r.get("empty"))
        n_locs = _count_locations(records)
        lines.append(
            f"| {name} | {len(records)} | {counts.get(0,0)} | "
            f"{counts.get(1,0)} | {counts.get(2,0)} | {n_empty} | {n_locs} |"
        )

    lines.append(f"")
    lines.append(f"## Per-Source Breakdown")
    lines.append(f"")
    lines.append(f"| Source | Train | Val | Test | Total |")
    lines.append(f"|--------|-------|-----|------|-------|")
    for src, stats in source_stats.items():
        lines.append(
            f"| {src} | {stats['train']} | {stats['val']} | {stats['test']} | {stats['total']} |"
        )

    lines.append(f"")
    lines.append(f"## Location Assignments (Val and Test)")
    lines.append(f"")
    lines.append(
        f"Train locations are omitted (majority). "
        f"Val and test location IDs are listed below for reproducibility."
    )
    lines.append(f"")

    all_sources = sorted(set(list(val_locs_by_source.keys()) + list(test_locs_by_source.keys())))
    for src in all_sources:
        lines.append(f"### {src}")
        lines.append(f"")
        lines.append(f"**Val:**")
        lines.append(f"```json")
        lines.append(json.dumps(sorted(val_locs_by_source.get(src, [])), indent=2))
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"**Test:**")
        lines.append(f"```json")
        lines.append(json.dumps(sorted(test_locs_by_source.get(src, [])), indent=2))
        lines.append(f"```")
        lines.append(f"")

    lines.append(f"## Notes")
    lines.append(f"")
    lines.append(
        f"- **Reusing location assignments**: pass `--location-assignments "
        f"{output_dir}/location_assignments.json` to reuse the same train/val/test "
        f"split when adding more images."
    )
    lines.append(
        f"- **Adding new data**: run `dataset_builder` with the same `--location-assignments` "
        f"path. New locations will be assigned randomly; existing locations keep their split."
    )
    lines.append(
        f"- **Empty images**: images with blank label files (empty=True) are included "
        f"as background/hard-negative examples."
    )
    lines.append(f"")

    readme_path = output_dir / "data_readme.md"
    readme_path.write_text("\n".join(lines) + "\n")
    LOGGER.info("README written: %s", readme_path)
    return readme_path


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    # per-source budgets
    wcs_max_animal: int = 800,
    wcs_max_vehicle: int = 50,
    wcs_max_empty: int = 200,
    cct_max_animal: int = 700,
    cct_max_empty: int = 200,
    serengeti_max_animal: int = 700,
    serengeti_max_empty: int = 200,
    island_max_animal: int = 500,
    island_max_empty: int = 150,
    coco_max_person: int = 500,
    coco_max_vehicle: int = 300,
    # source toggles
    skip_wcs: bool = False,
    skip_cct: bool = False,
    skip_serengeti: bool = False,
    skip_island: bool = False,
    skip_coco: bool = False,
    # split
    split: tuple = (0.80, 0.10, 0.10),
    seed: int = 42,
    num_download_workers: int = 8,
    location_assignments_path: Path | None = None,
    use_symlinks: bool = True,
    wcs_cache_dir: Path | None = None,
) -> Path:
    """Build the MegaDetector fine-tuning dataset.

    Downloads sources, merges records, applies location-aware splitting,
    installs the YOLO directory layout, writes the dataset YAML, and generates
    a provenance README.

    Returns the path to the dataset root (output_dir).
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {split}")

    # Import downloaders lazily (some have optional deps)
    from . import wcs_downloader
    from . import cct_downloader
    from . import serengeti_downloader
    from . import island_conservation_downloader
    from . import coco_downloader as _coco_dl

    raw_dir = output_dir / "_raw"

    # -----------------------------------------------------------------------
    # 1. Collect records from each source
    # -----------------------------------------------------------------------
    all_records: list[dict] = []

    if not skip_wcs:
        wcs_images = raw_dir / "wcs" / "images"
        wcs_labels = raw_dir / "wcs" / "labels"
        _wcs_cache = wcs_cache_dir or wcs_downloader.DEFAULT_CACHE_DIR
        est_gb = (wcs_max_animal + wcs_max_vehicle) * 1.5 / 1024
        LOGGER.info(
            "WCS: requesting up to %d animal images + %d vehicle images (~%.1f GB estimated).",
            wcs_max_animal, wcs_max_vehicle, est_gb,
        )
        wcs_records = wcs_downloader.download_and_convert_wcs(
            output_images_dir=wcs_images,
            output_labels_dir=wcs_labels,
            max_animal=wcs_max_animal,
            max_vehicle=wcs_max_vehicle,
            max_empty=wcs_max_empty,
            seed=seed,
            num_workers=num_download_workers,
            cache_dir=_wcs_cache,
        )
        LOGGER.info("WCS: %d usable records.", len(wcs_records))
        all_records.extend(wcs_records)
    else:
        LOGGER.info("Skipping WCS download (--skip-wcs).")

    if not skip_cct:
        cct_out = raw_dir / "cct"
        est_mb = cct_max_animal * 0.5
        LOGGER.info(
            "CCT: requesting up to %d animal images (~%.0f MB estimated).",
            cct_max_animal, est_mb,
        )
        cct_records = cct_downloader.download_and_convert_cct(
            output_dir=cct_out,
            max_animal=cct_max_animal,
            max_empty=cct_max_empty,
            seed=seed,
            num_workers=num_download_workers,
        )
        LOGGER.info("CCT: %d usable records.", len(cct_records))
        all_records.extend(cct_records)
    else:
        LOGGER.info("Skipping CCT download (--skip-cct).")

    if not skip_serengeti:
        ss_out = raw_dir / "serengeti"
        LOGGER.info(
            "Serengeti: requesting up to %d animal images.", serengeti_max_animal
        )
        ss_records = serengeti_downloader.download_and_convert_serengeti(
            output_dir=ss_out,
            max_animal=serengeti_max_animal,
            max_empty=serengeti_max_empty,
            seed=seed,
            num_workers=num_download_workers,
        )
        LOGGER.info("Serengeti: %d usable records.", len(ss_records))
        all_records.extend(ss_records)
    else:
        LOGGER.info("Skipping Serengeti download (--skip-serengeti).")

    if not skip_island:
        island_out = raw_dir / "island_conservation"
        LOGGER.info(
            "Island Conservation: requesting up to %d animal images.", island_max_animal
        )
        island_records = island_conservation_downloader.download_and_convert_island_conservation(
            output_dir=island_out,
            max_animal=island_max_animal,
            max_empty=island_max_empty,
            seed=seed,
            num_workers=num_download_workers,
        )
        LOGGER.info("Island Conservation: %d usable records.", len(island_records))
        all_records.extend(island_records)
    else:
        LOGGER.info("Skipping Island Conservation download (--skip-island).")

    if not skip_coco:
        coco_images = raw_dir / "coco" / "images"
        coco_labels = raw_dir / "coco" / "labels"
        est_mb = (coco_max_person + coco_max_vehicle) * 0.15
        LOGGER.info(
            "COCO: requesting up to %d person images + %d vehicle images (~%.0f MB estimated).",
            coco_max_person, coco_max_vehicle, est_mb,
        )
        try:
            raw_coco = _coco_dl.download_and_convert_coco(
                output_images_dir=coco_images,
                output_labels_dir=coco_labels,
                max_person=coco_max_person,
                max_vehicle=coco_max_vehicle,
                seed=seed,
            )
            # Wrap COCO records to add source/location_id/empty fields
            coco_records = [
                {
                    "image_path": r["image_path"],
                    "label_path": r["label_path"],
                    "source": "coco",
                    "location_id": None,
                    "empty": False,
                }
                for r in raw_coco
            ]
            LOGGER.info("COCO: %d usable records.", len(coco_records))
            all_records.extend(coco_records)
        except ImportError as exc:
            LOGGER.warning("Skipping COCO: %s", exc)
    else:
        LOGGER.info("Skipping COCO download (--skip-coco).")

    if not all_records:
        raise RuntimeError(
            "No records collected — all sources were skipped or produced no output. "
            "Check logs for errors."
        )

    LOGGER.info("Total records before split: %d", len(all_records))

    # -----------------------------------------------------------------------
    # 2. Separate records with / without location_id
    # -----------------------------------------------------------------------
    loc_records = [r for r in all_records if r.get("location_id")]
    no_loc_records = [r for r in all_records if not r.get("location_id")]

    LOGGER.info(
        "Records with location_id: %d  |  without: %d",
        len(loc_records), len(no_loc_records),
    )

    # -----------------------------------------------------------------------
    # 3. Load or compute location assignments
    # -----------------------------------------------------------------------
    assignments_path = output_dir / "location_assignments.json"

    if location_assignments_path and Path(location_assignments_path).exists():
        assignments = _load_location_assignments(Path(location_assignments_path))
        # Extend with any new locations not already assigned
        new_locs = {r["location_id"] for r in loc_records} - set(assignments)
        if new_locs:
            LOGGER.info("Assigning %d new locations randomly.", len(new_locs))
            new_sorted = sorted(new_locs)
            rng = random.Random(seed)
            rng.shuffle(new_sorted)
            n = len(new_sorted)
            n_train = int(n * split[0])
            n_val = int(n * split[1])
            for i, loc in enumerate(new_sorted):
                if i < n_train:
                    assignments[loc] = "train"
                elif i < n_train + n_val:
                    assignments[loc] = "val"
                else:
                    assignments[loc] = "test"
    else:
        assignments = _assign_locations(loc_records, split=split, seed=seed)

    _save_location_assignments(assignments_path, assignments)

    # -----------------------------------------------------------------------
    # 4. Apply split
    # -----------------------------------------------------------------------
    train_loc, val_loc, test_loc = _apply_location_split(loc_records, assignments, seed)

    if no_loc_records:
        train_nl, val_nl, test_nl = _split_records(no_loc_records, split, seed)
    else:
        train_nl, val_nl, test_nl = [], [], []

    train_r = train_loc + train_nl
    val_r = val_loc + val_nl
    test_r = test_loc + test_nl

    LOGGER.info(
        "Split: train=%d  val=%d  test=%d",
        len(train_r), len(val_r), len(test_r),
    )

    # -----------------------------------------------------------------------
    # 5. Install YOLO layout
    # -----------------------------------------------------------------------
    for split_name, records in [("train", train_r), ("val", val_r), ("test", test_r)]:
        LOGGER.info("Installing %s split …", split_name)
        _install_yolo_layout(records, split_name, output_dir, use_symlinks)

    # -----------------------------------------------------------------------
    # 6. Write YAML
    # -----------------------------------------------------------------------
    generate_dataset_yaml(
        output_dir,
        class_names=MD_CLASS_NAMES,
        yaml_name="megadetector_ft.yaml",
    )

    # -----------------------------------------------------------------------
    # 7. Write README
    # -----------------------------------------------------------------------
    _write_readme(output_dir, all_records, train_r, val_r, test_r, assignments)

    # -----------------------------------------------------------------------
    # 8. Print summary
    # -----------------------------------------------------------------------
    _print_summary(train_r, val_r, test_r, output_dir)

    LOGGER.info("Dataset ready.  Config: %s", output_dir / "megadetector_ft.yaml")
    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build MegaDetector fine-tuning dataset from multiple LILA camera trap "
            "datasets and COCO 2017.\n\n"
            "Downloads WCS, CCT, Serengeti, Island Conservation, and COCO subsets,\n"
            "merges them, performs a location-aware 80/10/10 train/val/test split,\n"
            "and lays out the YOLO directory structure."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, type=Path,
                   help="Root directory for the assembled dataset. (default: %(default)s)")

    # --- WCS ---
    g = p.add_argument_group("WCS")
    g.add_argument("--wcs-max-animal", default=800, type=int,
                   help="Max WCS animal images. ~1.5 MB each. (default: %(default)s)")
    g.add_argument("--wcs-max-vehicle", default=50, type=int,
                   help="Max WCS vehicle images. (default: %(default)s)")
    g.add_argument("--wcs-max-empty", default=200, type=int,
                   help="Max WCS background/empty images. (default: %(default)s)")

    # --- CCT ---
    g = p.add_argument_group("CCT")
    g.add_argument("--cct-max-animal", default=700, type=int,
                   help="Max CCT animal images. (default: %(default)s)")
    g.add_argument("--cct-max-empty", default=200, type=int,
                   help="Max CCT empty images. (default: %(default)s)")

    # --- Serengeti ---
    g = p.add_argument_group("Serengeti")
    g.add_argument("--serengeti-max-animal", default=700, type=int,
                   help="Max Serengeti animal images. (default: %(default)s)")
    g.add_argument("--serengeti-max-empty", default=200, type=int,
                   help="Max Serengeti empty images. (default: %(default)s)")

    # --- Island Conservation ---
    g = p.add_argument_group("Island Conservation")
    g.add_argument("--island-max-animal", default=500, type=int,
                   help="Max Island Conservation animal images. (default: %(default)s)")
    g.add_argument("--island-max-empty", default=150, type=int,
                   help="Max Island Conservation empty images. (default: %(default)s)")

    # --- COCO ---
    g = p.add_argument_group("COCO")
    g.add_argument("--coco-max-person", default=500, type=int,
                   help="Max COCO person images. ~150 KB each. (default: %(default)s)")
    g.add_argument("--coco-max-vehicle", default=300, type=int,
                   help="Max COCO vehicle images. (default: %(default)s)")

    # --- Source toggles ---
    g = p.add_argument_group("Source toggles")
    g.add_argument("--skip-wcs", action="store_true",
                   help="Skip WCS download.")
    g.add_argument("--skip-cct", action="store_true",
                   help="Skip CCT download.")
    g.add_argument("--skip-serengeti", action="store_true",
                   help="Skip Snapshot Serengeti download.")
    g.add_argument("--skip-island", action="store_true",
                   help="Skip Island Conservation download.")
    g.add_argument("--skip-coco", action="store_true",
                   help="Skip COCO download (requires fiftyone not installed).")

    # --- Split / misc ---
    g = p.add_argument_group("Split / misc")
    g.add_argument("--split", nargs=3, type=float, default=list(DEFAULT_SPLIT),
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Train / val / test fractions. Must sum to 1. (default: 0.80 0.10 0.10)")
    g.add_argument("--seed", default=42, type=int,
                   help="Random seed for reproducible sampling. (default: %(default)s)")
    g.add_argument("--workers", default=8, type=int,
                   help="Parallel HTTP workers. (default: %(default)s)")
    g.add_argument("--no-symlinks", action="store_true",
                   help="Copy files instead of symlinking them into the YOLO layout.")
    g.add_argument(
        "--location-assignments", default=None, type=Path, metavar="PATH",
        help="Path to a location_assignments.json from a previous run. "
             "Reuses existing location→split assignments and extends for new locations.",
    )
    g.add_argument("--log-level", default="INFO",
                   help="Logging verbosity. (default: INFO)")

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, args.log_level.upper(), logging.INFO),
    )
    split = tuple(args.split)
    if abs(sum(split) - 1.0) > 1e-6:
        logging.error(
            "ERROR: --split values %s do not sum to 1.0 (sum=%.4f)", args.split, sum(split)
        )
        raise SystemExit(1)

    build_dataset(
        output_dir=args.output_dir,
        wcs_max_animal=args.wcs_max_animal,
        wcs_max_vehicle=args.wcs_max_vehicle,
        wcs_max_empty=args.wcs_max_empty,
        cct_max_animal=args.cct_max_animal,
        cct_max_empty=args.cct_max_empty,
        serengeti_max_animal=args.serengeti_max_animal,
        serengeti_max_empty=args.serengeti_max_empty,
        island_max_animal=args.island_max_animal,
        island_max_empty=args.island_max_empty,
        coco_max_person=args.coco_max_person,
        coco_max_vehicle=args.coco_max_vehicle,
        skip_wcs=args.skip_wcs,
        skip_cct=args.skip_cct,
        skip_serengeti=args.skip_serengeti,
        skip_island=args.skip_island,
        skip_coco=args.skip_coco,
        split=split,
        seed=args.seed,
        num_download_workers=args.workers,
        location_assignments_path=args.location_assignments,
        use_symlinks=not args.no_symlinks,
    )
