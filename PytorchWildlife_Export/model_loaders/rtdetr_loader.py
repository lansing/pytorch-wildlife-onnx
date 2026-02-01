import os
import sys
import torch
import torch.nn as nn
import wget
from pathlib import Path
from collections import OrderedDict

from .base_loader import BaseModelLoader

class RTDETRLoader(BaseModelLoader):
    """
    A loader for the MegaDetectorV6 RT-DETR model using the rtdetrv2_pytorch library directly.
    """
    # Define model versions and their corresponding URLs/names
    MODEL_CONFIGS = {
        "MDV6-apa-rtdetr-c": {
            "url": "https://zenodo.org/records/15398270/files/MDV6-apa-rtdetr-c.pth?download=1",
            "model_name": "MDV6-apa-rtdetr-c.pth",
            "config_path_suffix": "rtdetrv2/rtdetrv2_r18vd_120e_megadetector.yml"
        },
        "MDV6-apa-rtdetr-e": {
            "url": "https://zenodo.org/records/15398270/files/MDV6-apa-rtdetr-e.pth?download=1",
            "model_name": "MDV6-apa-rtdetr-e.pth",
            "config_path_suffix": "rtdetrv2/rtdetrv2_r101vd_6x_megadetector.yml"
        }
    }

    def __init__(self, version='MDV6-apa-rtdetr-c', device="cpu"):
        if version not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported RT-DETR version: {version}. Choose from {list(self.MODEL_CONFIGS.keys())}")
        self.version = version
        self.device = device

    def load_model(self) -> nn.Module:
        """
        Loads the RT-DETR model directly using rtdetrv2_pytorch and returns the underlying PyTorch nn.Module.
        """
        config = self.MODEL_CONFIGS[self.version]
        model_url = config['url']
        model_filename = config['model_name']
        config_path_suffix = config['config_path_suffix']

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
        
        # --- Replicate logic from PytorchWildlife/models/detection/rtdetr_apache/rtdetr_apache_base.py ---
        # Need to add the rtdetrv2_pytorch path to sys.path to import YAMLConfig
        rtdetr_apache_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', 'CameraTraps', 'PytorchWildlife', 
            'models', 'detection', 'rtdetr_apache'
        ))
        if rtdetr_apache_path not in sys.path:
            sys.path.insert(0, rtdetr_apache_path)
            
        # Dynamically import YAMLConfig
        try:
            from rtdetrv2_pytorch.src.core import YAMLConfig
        except ImportError as e:
            raise ImportError(f"Could not import YAMLConfig. Ensure rtdetrv2_pytorch is correctly available. "
                              f"Attempted to add {rtdetr_apache_path} to sys.path. Error: {e}")

        # Construct the full config path
        rtdetrv2_pytorch_configs_path = os.path.join(rtdetr_apache_path, "rtdetrv2_pytorch", "configs")
        config_full_path = os.path.join(rtdetrv2_pytorch_configs_path, config_path_suffix)
        
        if not os.path.exists(config_full_path):
            raise FileNotFoundError(f"RT-DETR config file not found: {config_full_path}")

        # Load config and model
        cfg = YAMLConfig(config_full_path, resume=local_weights_path)
        
        checkpoint = torch.load(local_weights_path, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        cfg.model.load_state_dict(state)

        # Recreate the internal Model class structure for forward pass compatibility
        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()
                
            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        model = Model().to(self.device)
        print(f"Successfully loaded RT-DETR model (version: {self.version})")
        
        return model
