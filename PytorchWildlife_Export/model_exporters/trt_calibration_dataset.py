import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchvision.transforms.functional as F

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorch_wildlife_export" / "calibration"
DEFAULT_HF_DATASET = "lucabaggi/animal-wildlife"


class TRTCalibrationDataLoader:
    """
    Iterable that yields float32 NCHW image batches for INT8 calibration.

    Images are letterbox-resized to ``input_size × input_size``, normalised
    to [0, 1], and yielded as ``{"img": tensor}`` dicts where each tensor has
    shape ``(1, 3, H, W)`` and dtype float32.

    Used by ``calibrate_conv_nodes_scales`` (via ``wrap_nodes_in_int8_qdq``)
    to derive per-tensor activation scales for explicit INT8 quantization.
    The YOLO base model always expects float32 NCHW [0, 1] input, regardless
    of any input-preprocessing wrapper that may be merged later.

    Images are streamed from a HuggingFace dataset on first use and cached
    locally as a torch tensor list so repeated calibration runs skip the
    download.
    """

    def __init__(
        self,
        input_size: int,
        num_images: int = 100,
        hf_dataset: str = DEFAULT_HF_DATASET,
        hf_split: str = "train",
        cache_dir: Path = DEFAULT_CACHE_DIR,
    ):
        self.input_size = input_size
        self.num_images = num_images
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self.cache_dir = Path(cache_dir)
        self.batch_size = 1  # kept for legacy compatibility
        self._images: list[torch.Tensor] | None = None  # loaded lazily on first iter

    def _cache_path(self) -> Path:
        safe_name = self.hf_dataset.replace("/", "_")
        filename = (
            f"{safe_name}_{self.hf_split}_{self.num_images}"
            f"_{self.input_size}_nchw_uint8.pt"
        )
        return self.cache_dir / filename

    def _letterbox(self, pil_img) -> torch.Tensor:
        """
        Resize longest side to input_size and pad to a square.
        Returns a (3, H, W) uint8 tensor, values 0-255.
        """
        w, h = pil_img.size
        scale = self.input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(pil_img, (new_h, new_w))
        pad_w = self.input_size - new_w
        pad_h = self.input_size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        img = F.pad(img, padding, fill=0)
        arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W) uint8

    def _load_images(self) -> list[torch.Tensor]:
        cache_path = self._cache_path()

        if cache_path.exists():
            LOGGER.info(f"Loading calibration images from cache: {cache_path}")
            return torch.load(cache_path, weights_only=True)

        LOGGER.info(
            f"Streaming {self.num_images} calibration images from "
            f"'{self.hf_dataset}' (split='{self.hf_split}')..."
        )
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError(
                "The 'datasets' package is required for INT8 calibration. "
                "Install it with: pip install datasets"
            )

        ds = load_dataset(self.hf_dataset, split=self.hf_split, streaming=True)
        images = []
        for i, example in enumerate(ds):
            if i >= self.num_images:
                break
            pil_img = example["image"].convert("RGB")
            images.append(self._letterbox(pil_img))  # (3, H, W) uint8
            if (i + 1) % 100 == 0 or (i + 1) == self.num_images:
                LOGGER.info(f"  Processed {i + 1}/{self.num_images} calibration images")

        if len(images) < self.num_images:
            LOGGER.warning(
                f"Only {len(images)} images were available "
                f"(requested {self.num_images})."
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(images, cache_path)
        LOGGER.info(f"Calibration images cached to: {cache_path}")
        return images

    def __iter__(self) -> Iterator[dict]:
        if self._images is None:
            self._images = self._load_images()
        for tensor in self._images:
            # tensor: (3, H, W) uint8 — yield with batch dim
            # calibrate_conv_nodes_scales divides by 255 internally to
            # produce float32 [0, 1] matching the base YOLO model's input.
            yield {"img": tensor.unsqueeze(0).contiguous()}
