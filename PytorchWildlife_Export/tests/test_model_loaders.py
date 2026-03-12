import unittest
import os
import sys
import torch.nn as nn

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model_loaders.yolov9_loader import YoloV9Loader

class TestModelLoaders(unittest.TestCase):

    def test_yolov9_loader(self):
        """
        Test the YoloV9Loader to ensure it loads the model correctly.
        """
        print("\nTesting YoloV9Loader...")
        loader = YoloV9Loader(version='MDV6-yolov9-c')
        model = loader.load_model()
        self.assertIsInstance(model, nn.Module, "Loaded object should be a torch.nn.Module")
        print("YOLOv9 model loaded successfully and is an instance of torch.nn.Module.")



if __name__ == '__main__':
    unittest.main()