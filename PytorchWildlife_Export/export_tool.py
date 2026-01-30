import argparse
import os
import sys

# Add the project's top-level directory to the Python path
project_top_level = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_top_level not in sys.path:
    sys.path.insert(0, project_top_level)

from PytorchWildlife_Export.model_loaders.yolov9_loader import YoloV9Loader
from PytorchWildlife_Export.model_loaders.rtdetr_loader import RTDETRLoader
from PytorchWildlife_Export.model_exporters.yolov9_onnx_exporter import YoloV9ONNXExporter
from PytorchWildlife_Export.model_exporters.rtdetr_onnx_exporter import RTDETRONNXExporter
from PytorchWildlife_Export.model_exporters.yolov10_v9_compatible_exporter import YOLOv10V9CompatibleONNXExporter # Re-add import

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Wildlife models to ONNX format.")

    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["yolov9", "rtdetr", "yolov10_v9_compatible"], # Re-add yolov10_v9_compatible
        help="Type of the model to export (e.g., 'yolov9', 'rtdetr', 'yolov10_v9_compatible')."
    )
    parser.add_argument(
        "--model_version",
        type=str,
        required=True,
        help="Specific version of the model (e.g., 'MDV6-yolov9-c', 'MDV6-apa-rtdetr-c', 'MDV6-yolov10-e')."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the ONNX model will be saved (e.g., 'exported_models/my_model.onnx')."
    )
    parser.add_argument(
        "--format",
        type=str,
        default="float32",
        choices=["float32", "float16", "int8"],
        help="Numeric format for the exported ONNX model ('float32', 'float16', 'int8')."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version to use for export. Default is 18."
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Enable ONNX graph simplification using onnx-simplifier."
    )
    parser.add_argument(
        "--input_img_size",
        type=int,
        default=None,
        help="Square input image size (e.g., 1280 for 1280x1280). Required for some models. Overrides default."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load the PyTorch model on (e.g., 'cpu', 'cuda:0'). Default is 'cpu'."
    )

    args = parser.parse_args()

    # --- Load Model ---
    model_loader = None
    if args.model_type == "yolov9" or args.model_type == "yolov10_v9_compatible": # YOLOv9Loader also handles YOLOv10 pt models
        model_loader = YoloV9Loader(version=args.model_version, device=args.device)
    elif args.model_type == "rtdetr":
        model_loader = RTDETRLoader(version=args.model_version, device=args.device)
    else:
        print(f"Error: Unsupported model_type '{args.model_type}'.")
        sys.exit(1)
    
    print(f"Loading PyTorch model '{args.model_version}'...")
    model_pt = model_loader.load_model()
    print("PyTorch model loaded successfully.")

    # --- Determine Input Shape ---
    input_shape = None
    if args.model_type == "yolov9" or args.model_type == "yolov10_v9_compatible":
        # For ultralytics YOLO models, input_img_size directly maps to imgsz for export
        if args.input_img_size:
            input_shape = (1, 3, args.input_img_size, args.input_img_size)
        else:
            # Use default from YoloV9ONNXExporter if not specified
            input_shape = (1, 3, 1280, 1280) 
    elif args.model_type == "rtdetr":
        if args.input_img_size:
            input_shape = (1, 3, args.input_img_size, args.input_img_size)
        else:
            # Use default from RTDETRONNXExporter if not specified
            input_shape = (1, 3, 640, 640)
    
    if input_shape is None:
        print("Error: Input image size could not be determined. Please specify --input_img_size.")
        sys.exit(1)

    # --- Export Model ---
    exported_path = None
    if args.model_type == "yolov9":
        model_exporter = YoloV9ONNXExporter()
        exported_path = model_exporter.export(
            model=model_pt, # YOLO object
            output_path=args.output_path,
            input_shape=input_shape, # Passed to influence imgsz in ultralytics.YOLO.export
            opset_version=args.opset,
            do_simplify=args.simplify,
            export_format=args.format,
            nms=False # Force NMS to False for the raw output export
        )
    elif args.model_type == "yolov10_v9_compatible":
        model_exporter = YOLOv10V9CompatibleONNXExporter()
        exported_path = model_exporter.export(
            model=model_pt, # YOLO object
            output_path=args.output_path,
            input_shape=input_shape, # Passed to influence imgsz in ultralytics.YOLO.export
            opset_version=args.opset,
            do_simplify=args.simplify,
            export_format=args.format,
            num_classes=len(model_pt.model.names) if hasattr(model_pt.model, 'names') else 3 # Infer num_classes
        )
    elif args.model_type == "rtdetr":
        model_exporter = RTDETRONNXExporter()
        exported_path = model_exporter.export(
            model=model_pt, # nn.Module object
            output_path=args.output_path,
            input_shape=input_shape, # Passed for dummy input creation
            opset_version=args.opset,
            do_simplify=args.simplify,
            export_format=args.format
            # NMS argument is not applicable for RT-DETR via torch.onnx.export in this context
        )
    
    if exported_path:
        print(f"Model successfully exported to: {exported_path}")
    else:
        print("Model export failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()