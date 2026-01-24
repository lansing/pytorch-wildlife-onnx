import onnxruntime as ort
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ONNXModelValidator:
    """
    A utility class to load, inspect, and validate ONNX models.
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.input_names = []
        self.output_names = []
        self.input_shapes = {}
        self.output_shapes = {}
        self.opset_version = None

    def load_model(self):
        """
        Loads the ONNX model using onnxruntime.
        """
        try:
            self.session = ort.InferenceSession(self.model_path, providers=ort.get_available_providers())
            self._get_model_metadata()
            logging.info(f"Successfully loaded ONNX model from: {self.model_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to load ONNX model from {self.model_path}: {e}")
            self.session = None
            return False

    def _get_model_metadata(self):
        """
        Extracts input, output, and opset metadata from the loaded ONNX model.
        """
        if not self.session:
            logging.warning("Model not loaded. Cannot extract metadata.")
            return

        # Input metadata
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.input_shapes = {inp.name: inp.shape for inp in self.session.get_inputs()}

        # Output metadata
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.output_shapes = {out.name: out.shape for out in self.session.get_outputs()}

        # Opset version (this information is not directly available from ort.InferenceSession)
        # It usually needs to be parsed from the ONNX graph itself or derived from the model file.
        # For now, we'll skip opset_version as it's not directly exposed by InferenceSession
        # and focus on input/output details which are critical for validation.
        logging.info("Extracted model metadata.")

    def get_model_info(self):
        """
        Returns a dictionary containing key model information.
        """
        if not self.session:
            return {"status": "Model not loaded."}
        
        return {
            "status": "Model loaded successfully",
            "model_path": self.model_path,
            "input_names": self.input_names,
            "input_shapes": self.input_shapes,
            "output_names": self.output_names,
            "output_shapes": self.output_shapes,
            # "opset_version": self.opset_version, # Temporarily omitted
            "providers": self.session.get_providers()
        }

    def validate_forward_pass(self, input_data: dict = None):
        """
        Performs a forward pass with dummy data or provided input data to validate the model.
        
        Args:
            input_data (dict, optional): A dictionary mapping input names to numpy arrays.
                                        If None, dummy data is generated based on input shapes.

        Returns:
            bool: True if the forward pass is successful, False otherwise.
            dict: The output of the model if successful, otherwise an empty dict.
        """
        if not self.session:
            logging.error("Model not loaded. Cannot perform forward pass.")
            return False, {}

        if not self.input_names:
            logging.error("No input names found. Cannot generate dummy input.")
            return False, {}

        if input_data is None:
            # Generate dummy input data
            dummy_input = {}
            for name, shape in self.input_shapes.items():
                # Replace dynamic dimensions (e.g., 'N' or -1) with a fixed value (e.g., 1)
                concrete_shape = [s if isinstance(s, int) and s != -1 else 1 for s in shape]
                # Assuming float32 for dummy input, common for many models
                dummy_input[name] = np.random.rand(*concrete_shape).astype(np.float32)
            input_feed = dummy_input
            logging.info("Generated dummy input data for forward pass.")
        else:
            input_feed = input_data
            logging.info("Using provided input data for forward pass.")

        try:
            outputs = self.session.run(self.output_names, input_feed)
            output_dict = dict(zip(self.output_names, outputs))
            logging.info("Forward pass successful.")
            return True, output_dict
        except Exception as e:
            logging.error(f"Forward pass failed: {e}")
            return False, {}

if __name__ == '__main__':
    # Example Usage:
    # This block will only run if the script is executed directly, not when imported.
    # For actual testing, use the test_onnx_validator.py script.

    # This path needs to be adjusted based on where you run the script from
    # Assuming script is run from project root, and sample_models is a sibling directory
    model_file = "../../sample_models/MDV6-yolov9-c-320-16b.onnx"
    
    validator = ONNXModelValidator(model_file)
    if validator.load_model():
        model_info = validator.get_model_info()
        print("\nModel Info:")
        for k, v in model_info.items():
            print(f"  {k}: {v}")

        print("\nAttempting forward pass with dummy data...")
        success, outputs = validator.validate_forward_pass()
        if success:
            print("Forward pass completed successfully.")
            for name, output_array in outputs.items():
                print(f"  Output '{name}' shape: {output_array.shape}, Dtype: {output_array.dtype}")
                # print(f"  Output '{name}' sample: {output_array.flatten()[:5]}") # Print first 5 elements
        else:
            print("Forward pass failed.")
    else:
        print("Model loading failed. Validator cannot proceed.")
