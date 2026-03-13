"""
sweep_eval.py
-------------
Evaluate all standard MDV6-yolov10 variants and collect metrics into a CSV.

Iterates the same variant matrix as sweep_export.py.  Missing model files are
skipped with a warning.  Metrics are printed as each model completes, then a
full CSV is written at the end.

Usage
-----
    # inside the TRT Docker container (required for TRT engines):
    python -m PytorchWildlife_Export.sweep_eval \\
        --models-dir /exported_models \\
        --dataset    /data/md_ft/megadetector_ft.yaml \\
        --split      val \\
        --out        /exported_models/eval_results.csv

    # host shortcut — see `make sweep-eval`
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import product
from pathlib import Path

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from PytorchWildlife_Export.naming import build_output_filename
from PytorchWildlife_Export.dataset.eval import run_eval

# ---------------------------------------------------------------------------
# Variant dimensions — must stay in sync with sweep_export.py
# ---------------------------------------------------------------------------

MODEL_VERSIONS = ["MDV6-yolov10-e", "MDV6-yolov10-c"]
INPUT_SIZES    = [640, 320]
FORMATS        = ["float16", "int8"]
RUNTIMES       = ["onnx", "tensorrt"]
MODEL_TYPE     = "yolov10"

# CSV column order
_CLASSES  = ["animal", "person", "vehicle"]
_METRICS  = ["AP50", "AP50_95", "AR50", "AR50_95"]

CSV_FIELDS = (
    ["model", "model_version", "format", "size", "runtime", "n_images"]
    + [f"{cls}_{m}" for cls in _CLASSES for m in _METRICS]
    + ["mAP50", "mAP50_95", "mAR50", "mAR50_95"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_variants(models_dir: Path) -> list[tuple[dict, Path]]:
    """Return (config, model_path) for every variant, same exclusions as export."""
    out = []
    for version, size, fmt, runtime in product(MODEL_VERSIONS, INPUT_SIZES, FORMATS, RUNTIMES):
        if runtime == "onnx" and fmt == "int8":
            continue  # not exported — skip
        filename = build_output_filename(
            model_version=version,
            format=fmt,
            input_img_size=size,
            model_type=MODEL_TYPE,
            denormalized_input=True,
            nhwc_input=True,
            uint8_input=True,
            runtime=runtime,
        )
        out.append((
            {"model_version": version, "format": fmt, "size": size, "runtime": runtime},
            models_dir / filename,
        ))
    return out


def _results_to_csv_row(cfg: dict, model_path: Path, results: dict) -> dict:
    row: dict = {
        "model":         model_path.name,
        "model_version": cfg["model_version"],
        "format":        cfg["format"],
        "size":          cfg["size"],
        "runtime":       cfg["runtime"],
        "n_images":      results["n_images"],
    }
    for cls in _CLASSES:
        stats = results["per_class"].get(cls, {})
        for m in _METRICS:
            row[f"{cls}_{m}"] = f"{stats.get(m, 0.0):.4f}"
    for key in ("mAP50", "mAP50_95", "mAR50", "mAR50_95"):
        row[key] = f"{results.get(key, 0.0):.4f}"
    return row


def _print_progress_header() -> None:
    # Compact one-liner header printed before the sweep starts
    cols = f"{'Model':<52}  {'anAP50':>7}  {'anAR50':>7}  {'mAP50':>7}  {'mAR50':>7}"
    print(f"\n{cols}")
    print("-" * len(cols))


def _print_progress_row(cfg: dict, model_path: Path, results: dict) -> None:
    an   = results["per_class"].get("animal", {})
    name = model_path.name
    print(
        f"{name:<52}  "
        f"{an.get('AP50', 0):.4f}  "
        f"{an.get('AR50', 0):.4f}  "
        f"{results['mAP50']:.4f}  "
        f"{results['mAR50']:.4f}"
    )


def _print_final_table(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n" + "=" * 100)
    print("  SWEEP EVAL SUMMARY")
    print("=" * 100)
    hdr = (
        f"  {'Model':<52}  {'anAP50':>7}  {'anAR50':>7}  "
        f"{'perAP50':>7}  {'vehAP50':>7}  {'mAP50':>7}  {'mAR50':>7}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in rows:
        print(
            f"  {row['model']:<52}  "
            f"{row['animal_AP50']:>7}  {row['animal_AR50']:>7}  "
            f"{row['person_AP50']:>7}  {row['vehicle_AP50']:>7}  "
            f"{row['mAP50']:>7}  {row['mAR50']:>7}"
        )
    print("=" * 100 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep-eval all standard MDV6-yolov10 variants and write a metrics CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models-dir", default="/exported_models", metavar="DIR",
        help="Directory containing exported model files. (default: /exported_models)",
    )
    parser.add_argument(
        "--dataset", required=True, metavar="YAML",
        help="Path to megadetector_ft.yaml dataset config.",
    )
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "test"],
        help="Dataset split to evaluate. (default: val)",
    )
    parser.add_argument(
        "--out", default=None, metavar="CSV",
        help="Output CSV path. (default: <models-dir>/eval_results_<split>.csv)",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="VERSION",
        default=MODEL_VERSIONS, choices=MODEL_VERSIONS,
        help="Subset of model versions to evaluate.",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, metavar="PX",
        default=INPUT_SIZES,
        help="Subset of input sizes to evaluate.",
    )
    parser.add_argument(
        "--formats", nargs="+", metavar="FMT",
        default=FORMATS, choices=FORMATS,
        help="Subset of formats to evaluate.",
    )
    parser.add_argument(
        "--runtimes", nargs="+", metavar="RT",
        default=RUNTIMES, choices=RUNTIMES,
        help="Subset of runtimes to evaluate.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.001, metavar="THR",
        help="Confidence threshold for eval. (default: 0.001)",
    )
    parser.add_argument(
        "--max-images", type=int, default=None, metavar="N",
        help="Evaluate only the first N images per model (quick smoke test).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print the full per-model eval table in addition to the compact sweep row.",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for eval. (default: WARNING — suppress per-image noise)",
    )
    args = parser.parse_args(argv)

    import logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    models_dir = Path(args.models_dir)
    out_csv    = Path(args.out) if args.out else models_dir / f"eval_results_{args.split}.csv"

    all_variants = _all_variants(models_dir)
    variants = [
        (cfg, path) for cfg, path in all_variants
        if cfg["model_version"] in args.models
        and cfg["size"] in args.sizes
        and cfg["format"] in args.formats
        and cfg["runtime"] in args.runtimes
    ]

    total   = len(variants)
    skipped = 0
    failed  = 0
    csv_rows: list[dict] = []

    print(f"\nSweep eval: {total} variant(s)  split={args.split}  dataset={args.dataset}")
    _print_progress_header()

    for i, (cfg, model_path) in enumerate(variants, 1):
        label  = model_path.name
        prefix = f"[{i}/{total}]"

        if not model_path.exists():
            print(f"{prefix} SKIP (not found)  {label}")
            skipped += 1
            continue

        print(f"{prefix} evaluating  {label} …", end="", flush=True)
        try:
            results = run_eval(
                model_path=str(model_path),
                dataset_yaml=args.dataset,
                split=args.split,
                confidence_threshold=args.conf,
                max_images=args.max_images,
                quiet=not args.verbose,
            )
        except Exception as exc:
            print(f"\r{prefix} FAILED  {label}  — {exc}")
            failed += 1
            continue

        # Overwrite the "evaluating…" line with the result
        print(f"\r", end="")
        _print_progress_row(cfg, model_path, results)

        csv_rows.append(_results_to_csv_row(cfg, model_path, results))

    _print_final_table(csv_rows)

    # Write CSV
    if csv_rows:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"CSV written → {out_csv}")
    else:
        print("No results to write.")

    print(f"\nDone.  evaluated={len(csv_rows)}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
