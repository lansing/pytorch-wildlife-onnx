import argparse
import os
import sys

import torch

# Add the project's top-level directory to the Python path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

from PytorchWildlife_Export.model_exporters.input_preprocessing_wrapper import (
    InputPreprocessingWrapper,
)
# from PytorchWildlife_Export.model_exporters.onnx_exporter import ONNXExporter
# from PytorchWildlife_Export.model_exporters.rtdetr_onnx_exporter import (
#     RTDETRONNXExporter,
# )
# from PytorchWildlife_Export.model_exporters.yolov9_onnx_exporter import (
#     YoloV9ONNXExporter,
# )
from PytorchWildlife_Export.model_exporters.yolov10_v9_compatible_exporter import (
    YOLOv10V9CompatibleONNXExporter,
)
from PytorchWildlife_Export.model_loaders.rtdetr_loader import RTDETRLoader
from PytorchWildlife_Export.model_loaders.yolov9_loader import YoloV9Loader


def export_with_preprocessing(
    model_pt,
    model_type: str,
    input_shape: tuple,
    output_path: str,
    opset: int,
    do_simplify: bool,
    export_format: str,
    allow_denormalized: bool,
    allow_nhwc: bool,
    allow_uint8: bool,
    num_classes: int = 3,
    device: str = "cpu",
) -> str:
    """
    Export a model wrapped with InputPreprocessingWrapper using torch.onnx.export.

    This bypasses the ultralytics model.export() path so that we can supply a
    correctly-typed/shaped dummy input (NHWC or uint8) for accurate tracing.
    The preprocessing ops therefore appear in the ONNX graph and are visible
    to the ONNX simplifier.

    Returns the path to the final exported ONNX model.
    """
    if model_type == "rtdetr":
        print(
            "Warning: input preprocessing options are not supported for RT-DETR models. "
            "Proceeding without preprocessing."
        )
        return None

    # --- Fuse layers (Conv+BN fusion, etc.) ---
    try:
        model_pt.fuse()
        print("Model layers fused.")
    except Exception as e:
        print(f"Note: model.fuse() skipped ({e}).")

    # --- Get the inner nn.Module ---
    inner_model = model_pt.model
    inner_model.eval()

    # --- For v9-compat, prepend the output converter so it is part of the traced graph ---
    if model_type == "yolov10_v9_compatible":
        from PytorchWildlife_Export.model_exporters.yolov10_v9_wrapped_model import (
            YOLOv10V9WrappedModel,
        )

        inner_model = YOLOv10V9WrappedModel(inner_model, num_classes=num_classes)
        inner_model.eval()

    # --- Wrap with preprocessing ---
    wrapper = InputPreprocessingWrapper(
        inner_model=inner_model,
        allow_uint8=allow_uint8,
        allow_nhwc=allow_nhwc,
        allow_denormalized=allow_denormalized,
    )
    wrapper.eval()

    # --- Build dummy input ---
    torch_device = torch.device(device)
    dummy_input = wrapper.make_dummy_input(input_shape, device=torch_device)
    wrapper = wrapper.to(torch_device)

    # --- Determine paths (export float first, quantize after if needed) ---
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if export_format in ["int8", "uint8"]:
        temp_float_path = output_path.replace(".onnx", "_float.onnx")
        if allow_uint8:
            print(
                "Warning: combining --allow_uint8 with int8/uint8 export_format uses "
                "a float32 calibration dataset, which may not match the uint8 input "
                "the wrapped model declares. Static quantization may fail or produce "
                "inaccurate results."
            )
    else:
        temp_float_path = output_path

    # --- Export ---
    print(
        f"Exporting preprocessing-wrapped model to ONNX "
        f"(opset: {opset}, input shape: {dummy_input.shape}, dtype: {dummy_input.dtype})..."
    )
    torch.onnx.export(
        wrapper,
        dummy_input,
        temp_float_path,
        opset_version=opset,
    )

    print(f"Model exported to {temp_float_path}")

    # --- Quantize / simplify ---
    exporter = ONNXExporter()
    final_path = exporter._quantize_and_simplify_model(
        input_onnx_path=temp_float_path,
        output_final_onnx_path=output_path,
        export_format=export_format,
        do_simplify=do_simplify,
        input_shape=wrapper.get_input_shape(input_shape),
    )
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch Wildlife models to ONNX format."
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["yolov9", "rtdetr", "yolov10", "yolov10_v9_compatible"],
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
        choices=["float32", "float16", "int8", "uint8"],
        help="Numeric format for the exported ONNX model ('float32', 'float16', 'int8', 'uint8').",
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
        "--allow_denormalized",
        action="store_true",
        help="Add a /255 normalization layer so the model accepts 0-255 float input instead of 0.0-1.0.",
    )
    parser.add_argument(
        "--allow_nhwc",
        action="store_true",
        help="Add a transpose layer so the model accepts NHWC input instead of NCHW.",
    )
    parser.add_argument(
        "--allow_uint8",
        action="store_true",
        help="Change the model input dtype to uint8 and add a cast to float32 at the start.",
    )

    args = parser.parse_args()

    # --- Load Model ---
    model_loader = None
    if args.model_type in ("yolov9", "yolov10", "yolov10_v9_compatible"):
        model_loader = YoloV9Loader(version=args.model_version, device=args.device)
    elif args.model_type == "rtdetr":
        model_loader = RTDETRLoader(version=args.model_version, device=args.device)
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
            input_shape = (1, 3, 1280, 1280)
    elif args.model_type == "rtdetr":
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
    exported_path = None
    any_preprocessing = args.allow_denormalized or args.allow_nhwc or args.allow_uint8

    # # Use the wrapper-based export path for all YOLO types.
    # # RT-DETR preprocessing is not yet supported and will fall through with a warning.
    # num_classes = 3
    # if (
    #     args.model_type == "yolov10_v9_compatible"
    #     and hasattr(model_pt, "model")
    #     and hasattr(model_pt.model, "names")
    # ):
    #     num_classes = len(model_pt.model.names)

    # exported_path = export_with_preprocessing(
    #     model_pt=model_pt,
    #     model_type=args.model_type,
    #     input_shape=input_shape,
    #     output_path=args.output_path,
    #     opset=args.opset,
    #     do_simplify=args.simplify,
    #     export_format=args.format,
    #     allow_denormalized=args.allow_denormalized,
    #     allow_nhwc=args.allow_nhwc,
    #     allow_uint8=args.allow_uint8,
    #     num_classes=num_classes,
    #     device=args.device,
    # )

    # TODO try applying preprocessing wrapper here
    # and fix input_shape

    # --- Wrap with preprocessing ---
    preprocessor = InputPreprocessingWrapper(
        allow_uint8=args.allow_uint8,
        allow_nhwc=args.allow_nhwc,
        allow_denormalized=args.allow_denormalized,
    )

    # # --- Build dummy input ---
    # torch_device = torch.device(args.device)
    # dummy_input = preprocessor.make_dummy_input(input_shape, device=torch_device)
    # # TODO this is gonna f up the original v9 and v10 exports if we don't add preprocessor to those
    # input_shape = dummy_input.shape

    if args.model_type in ("yolov9", "yolov10"):
        model_exporter = YoloV9ONNXExporter()
        exported_path = model_exporter.export(
            model=model_pt,
            output_path=args.output_path,
            input_shape=input_shape,
            opset_version=args.opset,
            do_simplify=args.simplify,
            export_format=args.format,
            nms=False,
        )
    elif args.model_type == "yolov10_v9_compatible":
        model_exporter = YOLOv10V9CompatibleONNXExporter()
        exported_path = model_exporter.export(
            model=model_pt,
            output_path=args.output_path,
            input_shape=input_shape,
            opset_version=args.opset,
            do_simplify=args.simplify,
            export_format=args.format,
            num_classes=len(model_pt.model.names)
            if hasattr(model_pt.model, "names")
            else 3,
            allow_uint8=args.allow_uint8,
            allow_nhwc=args.allow_nhwc,
            allow_denormalized=args.allow_denormalized
        )
    # elif args.model_type == "rtdetr":
    #     model_exporter = RTDETRONNXExporter()
    #     exported_path = model_exporter.export(
    #         model=model_pt,
    #         output_path=args.output_path,
    #         input_shape=input_shape,
    #         opset_version=args.opset,
    #         do_simplify=args.simplify,
    #         export_format=args.format,
    #     )
    #
    if exported_path:
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

    else:
        print("Model export failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
