"""
PytorchWildlife_Export.dataset
==============================
Utilities for building a MegaDetector fine-tuning dataset from LILA WCS Camera
Traps (animals + vehicles) and COCO 2017 (people + vehicles), and for evaluating
exported ONNX / TensorRT models against that dataset.

Typical programmatic usage::

    from PytorchWildlife_Export.dataset import build_dataset, run_eval

    yaml_path = build_dataset(output_dir=Path("data/md_ft"))
    results   = run_eval("exported_models/model.onnx", dataset_yaml=str(yaml_path))
    print(results["mAP50"], results["mAP50_95"])
"""

from .dataset_builder import build_dataset
from .eval import run_eval

__all__ = ["build_dataset", "run_eval"]
