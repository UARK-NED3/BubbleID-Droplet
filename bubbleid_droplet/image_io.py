from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if image.ndim == 2:
        image = cv2.cvtColor(_to_uint8(image), cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(_to_uint8(image), cv2.COLOR_BGRA2BGR)
    else:
        image = _to_uint8(image)
    return image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return cv2.convertScaleAbs(image, alpha=255.0 / max(float(image.max()), 1.0))
    image = image.astype(np.float32)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))
    if max_val <= min_val:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image - min_val) * (255.0 / (max_val - min_val))).clip(0, 255).astype(np.uint8)


def write_mask_png(path: str | Path, label_mask: np.ndarray) -> None:
    binary = label_mask > 0
    rgb = np.zeros((*binary.shape, 3), dtype=np.uint8)
    rgb[~binary] = [128, 0, 128]
    rgb[binary] = [255, 255, 0]
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
