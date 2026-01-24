import unittest
import os
import sys
import numpy as np

# Add the project root to the Python path to allow importing our modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from model_validators.onnx_validator import ONNXModelValidator

class TestONNXModelValidator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # The model path is relative to the root of the project.
        cls.model_path = os.path.join(
            os.path.dirname(project_root), 
            'sample_models', 
            'MDV6-yolov9-c-320-16b.onnx'
        )
        
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(
                f"Test model not found at: {cls.model_path}. "
                "Ensure the 'sample_models' directory is correctly placed relative to the project root."
            )

        cls.validator = ONNXModelValidator(cls.model_path)

    def test_01_load_model(self):
        """
        Test if the ONNX model can be loaded successfully.
        """
        print(f"Attempting to load model from: {self.model_path}")
        is_loaded = self.validator.load_model()
        self.assertTrue(is_loaded, "Model should be loaded successfully.")
        self.assertIsNotNone(self.validator.session, "Session should not be None after loading.")

    def test_02_get_model_info(self):
        """
        Test if model metadata can be extracted correctly.
        Assumes the model is already loaded from test_01.
        """
        if not self.validator.session:
            self.validator.load_model()

        model_info = self.validator.get_model_info()
        
        self.assertEqual(model_info['status'], "Model loaded successfully")
        
        # Check for expected keys
        self.assertIn('input_names', model_info)
        self.assertIn('input_shapes', model_info)
        self.assertIn('output_names', model_info)
        self.assertIn('output_shapes', model_info)

        # For this specific YOLO-based model, we expect one input and at least one output.
        self.assertEqual(len(model_info['input_names']), 1, "Should have exactly one input.")
        self.assertGreaterEqual(len(model_info['output_names']), 1, "Should have at least one output.")

        # Check input shape format (e.g., [1, 3, 320, 320])
        input_name = model_info['input_names'][0]
        input_shape = model_info['input_shapes'][input_name]
        self.assertEqual(len(input_shape), 4, "Input should be a 4D tensor (N, C, H, W).")
        self.assertEqual(input_shape[1], 3, "Input should have 3 channels (RGB).")
        # Height and width can be dynamic ('dim' or number)
        self.assertTrue(isinstance(input_shape[2], (int, str)), "Height should be an int or string.")
        self.assertTrue(isinstance(input_shape[3], (int, str)), "Width should be an int or string.")


    def test_03_validate_forward_pass_dummy_data(self):
        """
        Test the forward pass with automatically generated dummy data.
        """
        if not self.validator.session:
            self.validator.load_model()

        success, outputs = self.validator.validate_forward_pass()
        
        self.assertTrue(success, "Forward pass with dummy data should succeed.")
        self.assertIsInstance(outputs, dict, "Outputs should be a dictionary.")
        self.assertGreater(len(outputs), 0, "Outputs should not be empty.")

        # Check that the output names match the model's output names
        self.assertEqual(set(outputs.keys()), set(self.validator.output_names))

        # Check the output type and content
        first_output_name = self.validator.output_names[0]
        first_output_array = outputs[first_output_name]
        self.assertIsInstance(first_output_array, np.ndarray, "Output should be a numpy array.")
        self.assertGreater(first_output_array.size, 0, "Output array should not be empty.")

if __name__ == '__main__':
    unittest.main()
