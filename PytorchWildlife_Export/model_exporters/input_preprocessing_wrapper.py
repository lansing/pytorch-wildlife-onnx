import torch
import torch.nn as nn


@torch.jit.script
def transpose_input(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] > x.shape[3]:
        return x.permute(0, 3, 1, 2)
    return x


class InputPreprocessingWrapper(nn.Module):
    """
    Wraps a PyTorch detection model with optional input preprocessing ops.

    This wrapper is applied before ONNX export so that the preprocessing
    steps become part of the exported ONNX graph and are visible to the
    ONNX simplifier.

    Operations applied in forward (in order):
      1. Cast uint8 → float32  (if allow_uint8)
      2. Transpose NHWC → NCHW (if allow_nhwc)
      3. Divide by 255.0       (if allow_denormalized)

    Args:
        model: The wrapped detection model (nn.Module).
        allow_uint8: Accept uint8 input; cast to float32 before processing.
        allow_nhwc: Accept NHWC (N,H,W,C) input; transpose to NCHW.
        allow_denormalized: Accept 0-255 float input; scale to 0.0-1.0.
    """

    def __init__(
        self,
        allow_uint8: bool = False,
        allow_nhwc: bool = False,
        allow_denormalized: bool = False,
    ):
        super().__init__()
        self.allow_uint8 = allow_uint8
        self.allow_nhwc = allow_nhwc
        self.allow_denormalized = allow_denormalized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.allow_uint8:
            x = x.float()
        if self.allow_nhwc:
            x = x.permute(0, 3, 1, 2)
        if self.allow_denormalized:
            x = x / 255.0
        return x

    def get_input_shape(self, nchw_shape: tuple) -> tuple:
        """
        Given the NCHW shape the inner model expects, return the input shape
        this wrapper expects.

        Args:
            nchw_shape: e.g. (1, 3, 640, 640)

        Returns:
            (1, H, W, C) if allow_nhwc, else (1, C, H, W)
        """
        n, c, h, w = nchw_shape
        if self.allow_nhwc:
            return (n, h, w, c)
        return nchw_shape

    def get_input_dtype(self) -> torch.dtype:
        """Return the dtype this wrapper expects as its input."""
        return torch.uint8 if self.allow_uint8 else torch.float32

    def make_dummy_input(
        self, nchw_shape: tuple, device: torch.device = None
    ) -> torch.Tensor:
        """
        Create a dummy input tensor suitable for torch.onnx.export.

        Args:
            nchw_shape: The NCHW shape the inner model expects.
            device: Target device (defaults to CPU).

        Returns:
            A zero tensor with the correct shape and dtype.
        """
        if device is None:
            device = torch.device("cpu")
        shape = self.get_input_shape(nchw_shape)
        dtype = self.get_input_dtype()
        return torch.zeros(shape, dtype=dtype, device=device)
