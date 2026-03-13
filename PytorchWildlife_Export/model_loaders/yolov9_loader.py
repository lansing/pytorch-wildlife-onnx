import os
import sys
import torch.nn as nn
import wget
from ultralytics import YOLO
import torch 

from .base_loader import BaseModelLoader

class YoloV9Loader(BaseModelLoader):
    """
    A loader for the MegaDetectorV6 YOLOv9 model using the ultralytics library directly.
    """
    MODEL_CONFIGS = {
        'MDV6-yolov9-c': {
            'url': "https://zenodo.org/records/15398270/files/MDV6-yolov9-c.pt?download=1",
            'model_name': "MDV6-yolov9-c.pt",
            'imgsz': 1280
        },
        'MDV6-yolov9-e': {
            'url': "https://zenodo.org/records/15398270/files/MDV6-yolov9-e-1280.pt?download=1",
            'model_name': "MDV6-yolov9-e-1280.pt",
            'imgsz': 1280
        },
        'MDV6-yolov10-c': {
            'url': "https://zenodo.org/records/15398270/files/MDV6-yolov10-c.pt?download=1",
            'model_name': "MDV6-yolov10-c.pt",
            'imgsz': 1280
        },
        'MDV6-yolov10-e': {
            'url': "https://zenodo.org/records/15398270/files/MDV6-yolov10-e-1280.pt?download=1",
            'model_name': "MDV6-yolov10-e-1280.pt",
            'imgsz': 1280
        }
    }

    def __init__(self, version='MDV6-yolov9-c', device="cpu", weights: str | None = None):
        if version not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported YOLOv9 version: {version}. Choose from {list(self.MODEL_CONFIGS.keys())}")
        self.version = version
        self.device = device
        self.weights = weights  # optional local .pt override (e.g. QAT finetuned)

    def load_model(self) -> YOLO:
        """
        Loads the YOLOv9 model directly using ultralytics and returns the ultralytics.YOLO object.
        """
        if self.weights:
            # Use explicit local weights (e.g. QAT-finetuned checkpoint)
            print(f"Loading {self.version} from local weights: {self.weights}")
            model = YOLO(self.weights)
            model.to(self.device)
            print(f"Successfully loaded {self.version} from {self.weights}")
            return model

        config = self.MODEL_CONFIGS[self.version]
        model_url = config['url']
        model_filename = config['model_name']

        # Determine path to save/load weights
        weights_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
        os.makedirs(weights_dir, exist_ok=True)
        local_weights_path = os.path.join(weights_dir, model_filename)

        # Download weights if not present
        if not os.path.exists(local_weights_path):
            print(f"Downloading {self.version} weights from {model_url} to {local_weights_path}...")
            wget.download(model_url, out=local_weights_path)
            print("Download complete.")
        else:
            print(f"Using existing {self.version} weights from {local_weights_path}")

        # Load the model using ultralytics.YOLO
        model = YOLO(local_weights_path)
        model.to(self.device)
        print(f"Successfully loaded YOLOv9 model (version: {self.version})")

        # Return the YOLO object
        return model
