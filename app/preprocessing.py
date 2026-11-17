"""Deterministic image preprocessing: bytes -> normalized CHW float32 tensor."""
from __future__ import annotations

import io

import numpy as np
import torch
from PIL import Image

IMAGE_SIZE = 64
_PROBE_COLOR = (72, 128, 200)  # any opaque, deterministic RGB colour
_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def _to_tensor(img: Image.Image) -> torch.Tensor:
    img = img.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)


def preprocess(image_bytes: bytes) -> torch.Tensor:
    """Decode an image and return a (3, 64, 64) float32 tensor.

    Raises ValueError if the bytes are not a decodable image.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return _to_tensor(img)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a bad request
        raise ValueError(f"cannot decode image: {exc}") from exc


def recovery_probe_input() -> torch.Tensor:
    """One deterministic 64x64 RGB input for the recovery probe.

    It goes through the exact same normalisation as customer frames, so a
    successful probe proves the real input pipeline works, not a shortcut.
    Returns a (1, 3, 64, 64) batch tensor.
    """
    img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), _PROBE_COLOR)
    return _to_tensor(img).unsqueeze(0)
