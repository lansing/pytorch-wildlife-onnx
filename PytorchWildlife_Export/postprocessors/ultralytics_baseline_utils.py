from typing import List, Dict, Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results

def get_ultralytics_baseline_detections(
    onnx_model_path: str,
    image_path: str,
    confidence_threshold: float,
    iou_threshold: float,
    input_img_size: int,
    class_names: Dict[int, str]
) -> List[Dict]:
    """
    Retrieves detection results using ultralytics.YOLO.predict directly from an ONNX model.
    This serves as a baseline for comparison with custom post-processing logic.

    Args:
        onnx_model_path (str): Path to the ONNX model.
        image_path (str): Path to the input image.
        confidence_threshold (float): Confidence threshold for filtering detections.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression (NMS).
        input_img_size (int): Square input image size for inference.
        class_names (Dict[int, str]): Mapping of class IDs to class names.

    Returns:
        List[Dict]: A list of dictionaries, each representing a detected object.
                    Each dict contains 'box' (xyxy), 'confidence', 'class_id', 'class_name'.
    """
    try:
        yolo_model = YOLO(onnx_model_path)
    except Exception as e:
        print(f"Error loading YOLO model with ultralytics for baseline: {e}")
        return []

    results_list: List[Results] = yolo_model.predict(
        source=image_path,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=input_img_size,
        verbose=False
    )
    
    detections = []
    if results_list:
        for res in results_list: # Should be one result for one image
            for box_data in res.boxes:
                x1, y1, x2, y2 = [int(x) for x in box_data.xyxy[0].tolist()]
                confidence = float(box_data.conf[0])
                class_id = int(box_data.cls[0])
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_names.get(class_id, "unknown")
                })
    return detections
