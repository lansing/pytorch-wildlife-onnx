import argparse
import os
import sys
from pathlib import Path

import torch

# Add the project's top-level directory to the Python path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)
from PytorchWildlife_Export.model_exporters.yolov9_onnx_exporter import (
    YoloV9ONNXExporter,
)
from PytorchWildlife_Export.model_exporters.yolov10_v9_compatible_exporter import (
    YOLOv10V9CompatibleONNXExporter,
)
from PytorchWildlife_Export.model_loaders.yolov9_loader import YoloV9Loader


def parse_args(argv=None):
    """Parse export_tool arguments from *argv* (list of strings) or sys.argv if None."""
    parser = argparse.ArgumentParser(
        description="Export PyTorch Wildlife models to ONNX format."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["yolov9", "yolov10", "yolov10_v9_compatible"],
        help="Type of the model to export (e.g., 'yolov9', 'rtdetr', 'yolov10', 'yolov10_v9_compatible').",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="Specific version of the model (e.g., 'MDV6-yolov9-c', 'MDV6-apa-rtdetr-c', 'MDV6-yolov10-e').",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the ONNX model will be saved (e.g., 'exported_models/my_model.onnx').",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="float32",
        choices=["float32", "float16", "int8"],
        help="Numeric format for the exported model ('float32', 'float16', 'int8').",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to use for export. Default is 18.",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Enable ONNX graph simplification using onnx-simplifier.",
    )
    parser.add_argument(
        "--input_img_size",
        type=int,
        default=None,
        help="Square input image size (e.g., 1280 for 1280x1280). Required for some models. Overrides default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the PyTorch model on (e.g., 'cpu', 'cuda:0'). Default is 'cpu'.",
    )

    # --- Input preprocessing options ---
    parser.add_argument(
        "--denormalized_input",
        action="store_true",
        help="Add a /255 normalization layer so the model accepts 0-255 float input instead of 0.0-1.0.",
    )
    parser.add_argument(
        "--nhwc_input",
        action="store_true",
        help="Add a transpose layer so the model accepts NHWC input instead of NCHW.",
    )
    parser.add_argument(
        "--uint8_input",
        action="store_true",
        help="Change the model input dtype to uint8 and add a cast to float32 at the start.",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default="onnx",
        choices=["onnx", "tensorrt"],
        help="Runtime target. Default is 'onnx'.",
    )
    parser.add_argument(
        "--num_calibration_images",
        type=int,
        default=100,
        help=(
            "Number of images to stream from the calibration dataset for INT8 "
            "quantization. Only used when --format int8. Default is 100."
        ),
    )
    parser.add_argument(
        "--quant_profile",
        type=str,
        default="blanket",
        choices=["conv", "blanket"],
        help=(
            "INT8 quantization profile. Controls which op types are wrapped in "
            "QDQ pairs. 'blanket' (default): Conv + Add + MaxPool — eliminates "
            "layout-reformat overhead on residual shortcut paths and enables "
            "Conv+SiLU+Add fusion as a single INT8 kernel in TensorRT; observed "
            "to be superior to 'conv' on all tested NVIDIA hardware. "
            "'conv': Conv only — retained for experimentation. "
            "Only used when --format int8."
        ),
    )

    return parser.parse_args(argv)


def main(args=None):
    if args is None:
        args = parse_args()

    # --- Guard: refuse to write model output inside the source package ---
    _output = Path(args.output_path).resolve()
    _pkg_root = Path(__file__).resolve().parent  # PytorchWildlife_Export/
    try:
        _output.relative_to(_pkg_root)
        # If no ValueError: the path is inside the package tree — reject it.
        print(
            f"Error: --output_path must not be inside '{_pkg_root}'.\n"
            f"  Use /exported_models or any path outside PytorchWildlife_Export."
        )
        sys.exit(1)
    except ValueError:
        pass  # Path is outside the package — allowed.

    # --- Load Model ---
    model_loader = None
    if args.model_type in ("yolov9", "yolov10", "yolov10_v9_compatible"):
        model_loader = YoloV9Loader(version=args.model_version, device=args.device)
    else:
        print(f"Error: Unsupported model_type '{args.model_type}'.")
        sys.exit(1)

    print(f"Loading PyTorch model '{args.model_version}'...")
    model_pt = model_loader.load_model()
    print("PyTorch model loaded successfully.")

    # --- Determine Input Shape (NCHW, as the inner model expects) ---
    input_shape = None
    if args.model_type in ("yolov9", "yolov10", "yolov10_v9_compatible"):
        if args.input_img_size:
            input_shape = (1, 3, args.input_img_size, args.input_img_size)
        else:
            input_shape = (1, 3, 640, 640)

    if input_shape is None:
        print(
            "Error: Input image size could not be determined. Please specify --input_img_size."
        )
        sys.exit(1)

    # --- Export Model ---
    exported_path = args.output_path
    os.makedirs(os.path.dirname(exported_path), exist_ok=True)

    num_classes = len(model_pt.model.names) if hasattr(model_pt.model, "names") else 3
    export_kwargs = dict(
        model=model_pt,
        output_path=args.output_path,
        input_shape=input_shape,
        opset_version=args.opset,
        do_simplify=args.simplify,
        export_format=args.format,
        num_classes=num_classes,
        uint8_input=args.uint8_input,
        nhwc_input=args.nhwc_input,
        denormalized_input=args.denormalized_input,
        runtime=args.runtime,
        num_calibration_images=args.num_calibration_images,
        model_type=args.model_type,
        quant_profile=args.quant_profile,
        device=args.device,
    )

    if args.model_type in ("yolov9", "yolov10"):
        model_exporter = YoloV9ONNXExporter()
        model_exporter.export(**export_kwargs)
    elif args.model_type == "yolov10_v9_compatible":
        model_exporter = YOLOv10V9CompatibleONNXExporter()
        model_exporter.export(**export_kwargs)

    print(f"Model successfully exported to: {exported_path}")

    # Write class names file
    class_names = ["animal", "person", "vehicle"]
    class_names_file_path = os.path.join(
        os.path.dirname(exported_path), "md.classes.txt"
    )
    try:
        with open(class_names_file_path, "w") as f:
            for name in class_names:
                f.write(name + "\n")
        print(f"Class names file created: {class_names_file_path}")
    except Exception as e:
        print(f"Error creating class names file: {e}")


if __name__ == "__main__":
    main()
