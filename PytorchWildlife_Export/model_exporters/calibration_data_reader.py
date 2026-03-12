from typing import Dict, Tuple

import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReader


class WildlifeCalibrationDataReader(CalibrationDataReader):
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        input_name: str,
        input_dtype: "str" = "float32",
        num_batches: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.num_batches = num_batches
        self.current_batch = 0
        self.input_name = input_name
        self.input_dtype = input_dtype

    def get_next(self) -> Dict[str, np.ndarray]:
        if self.current_batch < self.num_batches:
            # Generate dummy input data (e.g., random tensor)
            # This should match the expected input format of the model.
            if self.input_dtype == "uint8":
                input_data = np.random.randint(
                    0, 256, size=self.input_shape, dtype=np.uint8
                )
            else:
                input_data = np.random.randn(*self.input_shape).astype(np.float32)
            self.current_batch += 1
            return {self.input_name: input_data}
        else:
            return None

    def rewind(self):
        self.current_batch = 0

    def get_input_shape(self) -> Tuple[int, ...]:
        return self.input_shape
