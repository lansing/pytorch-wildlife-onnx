import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torchvision.transforms.functional as F

LOGGER = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pytorch_wildlife_export" / "calibration"
DEFAULT_HF_DATASET = "lucabaggi/animal-wildlife"
# TODO try w https://docs.ultralytics.com/datasets/detect/african-wildlife/#dataset-yaml


class TRTCalibrationDataLoader:
    """
    Iterable that yields {"img": tensor} dicts for TensorRT INT8 calibration.

    When used with the standard ultralytics EngineCalibrator (no preprocessing
    patch), tensors must be uint8 so that the calibrator's hard-coded /255.0
    produces float32 [0, 1].  When the _preprocessing_calibration_patch context
    manager is active, get_batch is replaced and tensors are forwarded as-is, so
    this loader must match the exact dtype/layout/range of the merged model's
    input binding:

      - baseline / nhwc_input only: uint8 CHW or HWC (patch divides by 255 → float32 0-1)
      - denormalized_input: float32 CHW or HWC, range 0-255
      - uint8_input: uint8 CHW or HWC, range 0-255

    Images are streamed from a HuggingFace dataset on first use and cached locally
    as a torch tensor list so repeated calibration runs skip the download.
    The cache key encodes all parameters that affect the tensor contents, so
    different configurations never share a cache file.
    """

    def __init__(
        self,
        input_size: int,
        num_images: int = 300,
        hf_dataset: str = DEFAULT_HF_DATASET,
        hf_split: str = "train",
        cache_dir: Path = DEFAULT_CACHE_DIR,
        nhwc_input: bool = False,
        uint8_input: bool = False,
        denormalized_input: bool = False,
    ):
        self.input_size = input_size
        self.num_images = num_images
        self.hf_dataset = hf_dataset
        self.hf_split = hf_split
        self.cache_dir = Path(cache_dir)
        self.nhwc_input = nhwc_input
        self.uint8_input = uint8_input
        self.denormalized_input = denormalized_input
        self.batch_size = 1  # EngineCalibrator reads this attribute
        self._images: list[torch.Tensor] | None = None  # loaded lazily on first iter

    def _cache_path(self) -> Path:
        safe_name = self.hf_dataset.replace("/", "_")
        layout = "nhwc" if self.nhwc_input else "nchw"
        if self.uint8_input:
            dtype = "uint8"
        elif self.denormalized_input:
            dtype = "float32_255"
        else:
            dtype = "uint8"  # baseline: uint8 for compensating-data trick
        filename = (
            f"{safe_name}_{self.hf_split}_{self.num_images}"
            f"_{self.input_size}_{layout}_{dtype}.pt"
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
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)

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
                "The 'datasets' package is required for INT8 TRT calibration. "
                "Install it with: pip install datasets"
            )

        ds = load_dataset(self.hf_dataset, split=self.hf_split, streaming=True)
        images = []
        for i, example in enumerate(ds):
            if i >= self.num_images:
                break
            pil_img = example["image"].convert("RGB")
            tensor = self._letterbox(pil_img)  # (3, H, W) uint8
            images.append(tensor)
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
            # tensor is (3, H, W) uint8 NCHW from _letterbox
            # 1. Layout
            if self.nhwc_input:
                tensor = tensor.permute(1, 2, 0).contiguous()  # CHW → HWC

            # 2. Dtype / range
            # uint8_input: keep uint8 0-255 (patched get_batch delivers as-is)
            # denormalized_input: float32 0-255 (patched get_batch delivers as-is)
            # baseline / nhwc_input only: keep uint8 — compensating-data trick:
            #   EngineCalibrator /255 → float32 0-1 (no patch active)
            if self.denormalized_input and not self.uint8_input:
                tensor = tensor.float()  # uint8 → float32, range still 0-255

            yield {"img": tensor.unsqueeze(0).contiguous()}
