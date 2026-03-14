"""
sweep_export.py
---------------
Export all standard permutations of the MDV6-yolov10 model family.

Variants swept
--------------
  Model  : MDV6-yolov10-e  MDV6-yolov10-c
  Size   : 640  320
  Format : float16  int8
  Runtime: onnx  tensorrt

All exports use the "all preprocessing" configuration:
  --denormalized_input --nhwc_input --uint8_input

INT8 TRT exports require calibration images; the dataset val split is used
by default (--calib-split val) so no extra data download is needed.

Usage
-----
    python PytorchWildlife_Export/sweep_export.py --output-dir exported_models
    python PytorchWildlife_Export/sweep_export.py --output-dir exported_models --dry-run
    python PytorchWildlife_Export/sweep_export.py --output-dir exported_models --skip-existing
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

# Ensure project root is on sys.path when run as a script
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from PytorchWildlife_Export.naming import build_output_filename

# ---------------------------------------------------------------------------
# Sweep dimensions
# ---------------------------------------------------------------------------

MODEL_VERSIONS = ["MDV6-yolov10-e", "MDV6-yolov10-c"]
INPUT_SIZES    = [640, 320]
FORMATS        = ["float16", "int8"]
RUNTIMES       = ["onnx", "tensorrt"]

# Fixed preprocessing flags applied to every export
PREPROC_FLAGS  = ["--denormalized_input", "--nhwc_input", "--uint8_input"]

MODEL_TYPE     = "yolov10"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_export_cmd(
    model_version: str,
    fmt: str,
    size: int,
    runtime: str,
    output_path: Path,
    num_calib_images: int,
    dataset_yaml: str | None,
    calib_split: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "PytorchWildlife_Export/export_tool.py",
        "--model_type",     MODEL_TYPE,
        "--model_version",  model_version,
        "--output_path",    str(output_path),
        "--format",         fmt,
        "--input_img_size", str(size),
        "--opset",          "18",
        "--simplify",
        "--runtime",        runtime,
        *PREPROC_FLAGS,
    ]
    if fmt == "int8":
        cmd += ["--quant_profile", "blanket"]
        cmd += ["--num_calibration_images", str(num_calib_images)]
        if runtime == "tensorrt" and dataset_yaml:
            cmd += ["--calibration_dataset", dataset_yaml,
                    "--calibration_split", calib_split]
    return cmd


def _all_variants(output_dir: Path) -> list[tuple[dict, Path]]:
    """Return list of (config_dict, output_path) for every permutation."""
    variants = []
    for version, size, fmt, runtime in product(MODEL_VERSIONS, INPUT_SIZES, FORMATS, RUNTIMES):
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
        variants.append(({
            "model_version": version,
            "format":        fmt,
            "size":          size,
            "runtime":       runtime,
        }, output_dir / filename))
    return variants


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep-export all standard MDV6-yolov10 variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="exported_models", metavar="DIR",
        help="Directory to write exported models. (default: exported_models)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip any variant whose output file already exists.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the commands that would be run without executing them.",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="VERSION",
        default=MODEL_VERSIONS,
        choices=MODEL_VERSIONS,
        help="Subset of model versions to export. (default: all)",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, metavar="PX",
        default=INPUT_SIZES,
        help="Subset of input sizes to export. (default: 640 320)",
    )
    parser.add_argument(
        "--formats", nargs="+", metavar="FMT",
        default=FORMATS,
        choices=FORMATS,
        help="Subset of formats to export. (default: float16 int8)",
    )
    parser.add_argument(
        "--runtimes", nargs="+", metavar="RT",
        default=RUNTIMES,
        choices=RUNTIMES,
        help="Subset of runtimes to export. (default: onnx tensorrt)",
    )
    parser.add_argument(
        "--num-calib-images", type=int, default=100, metavar="N",
        help="Calibration images for INT8 TRT exports. (default: 100)",
    )
    parser.add_argument(
        "--dataset-yaml", default=None, metavar="YAML",
        help="Dataset YAML for INT8 calibration (e.g. data/md_ft/megadetector_ft.yaml). "
             "If omitted, the exporter uses its built-in MegaDetector sample images.",
    )
    parser.add_argument(
        "--calib-split", default="val", choices=["train", "val", "test"],
        help="Split to use for INT8 calibration when --dataset-yaml is given. (default: val)",
    )
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build full variant list then filter to requested subset
    all_variants = _all_variants(output_dir)
    variants = [
        (cfg, path) for cfg, path in all_variants
        if cfg["model_version"] in args.models
        and cfg["size"] in args.sizes
        and cfg["format"] in args.formats
        and cfg["runtime"] in args.runtimes
    ]

    total = len(variants)
    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Sweep: {total} variant(s) to export\n")

    ok = skipped = failed = 0

    for i, (cfg, out_path) in enumerate(variants, 1):
        label = out_path.name
        prefix = f"[{i}/{total}]"

        if args.skip_existing and out_path.exists():
            print(f"{prefix} SKIP (exists)  {label}")
            skipped += 1
            continue

        cmd = _build_export_cmd(
            model_version=cfg["model_version"],
            fmt=cfg["format"],
            size=cfg["size"],
            runtime=cfg["runtime"],
            output_path=out_path,
            num_calib_images=args.num_calib_images,
            dataset_yaml=args.dataset_yaml,
            calib_split=args.calib_split,
        )

        print(f"{prefix} {'(dry) ' if args.dry_run else ''}EXPORT  {label}")
        if args.dry_run:
            print("  " + " ".join(cmd))
            ok += 1
            continue

        result = subprocess.run(cmd)
        if result.returncode == 0:
            print(f"{prefix} OK  {label}\n")
            ok += 1
        else:
            print(f"{prefix} FAILED (exit {result.returncode})  {label}\n")
            failed += 1

    print(f"\nDone.  ok={ok}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
