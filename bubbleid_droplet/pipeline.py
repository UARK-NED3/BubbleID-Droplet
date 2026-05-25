from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import regionprops

from .image_io import read_image, write_mask_png
from .predictors import ClassicalDropletPredictor, Detectron2BubbleIDPredictor, Prediction
from .tiling import iter_tiles


@dataclass
class RunSummary:
    image_path: Path
    output_dir: Path
    droplet_count: int
    mean_diameter_px: float
    std_diameter_px: float
    backend: str


def build_predictor(args):
    if args.backend == "classical":
        return ClassicalDropletPredictor(min_area=args.min_area)
    if not args.weights:
        raise ValueError("--weights is required when --backend detectron2")
    return Detectron2BubbleIDPredictor(
        weights=args.weights,
        score_threshold=args.score_threshold,
        device=args.device,
        min_size_test=args.min_size_test,
        max_size_test=args.max_size_test,
        detections_per_image=args.detections_per_image,
    )


def run_image(
    image_path: str | Path,
    output_dir: str | Path,
    predictor,
    backend: str,
    tile_size: int = 512,
    overlap: int = 128,
    iou_threshold: float = 0.30,
) -> RunSummary:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = read_image(image_path)
    h, w = image.shape[:2]

    predictions: list[Prediction] = []
    for tile in iter_tiles(h, w, tile_size=tile_size, overlap=overlap):
        crop = image[tile.y0 : tile.y1, tile.x0 : tile.x1]
        for pred in predictor(crop):
            global_mask = np.zeros((h, w), dtype=bool)
            global_mask[tile.y0 : tile.y1, tile.x0 : tile.x1] = pred.mask
            x0, y0, x1, y1 = pred.bbox
            predictions.append(
                Prediction(
                    mask=global_mask,
                    score=pred.score,
                    bbox=(x0 + tile.x0, y0 + tile.y0, x1 + tile.x0, y1 + tile.y0),
                )
            )

    kept = predictions if overlap == 0 else mask_nms(predictions, iou_threshold=iou_threshold)
    label_mask = masks_to_labels(kept, shape=(h, w))
    regions = regionprops(label_mask)
    diameters = np.array([equivalent_diameter(region) for region in regions], dtype=float)

    stem = image_path.stem
    write_mask_png(output_dir / f"{stem}_fullmask.png", label_mask)
    cv2.imwrite(str(output_dir / f"{stem}_labels.tif"), label_mask.astype(np.uint16))
    np.savetxt(output_dir / f"{stem}_diameters.txt", diameters)
    write_overlay(output_dir / f"{stem}_overlay.png", image, regions)

    return RunSummary(
        image_path=image_path,
        output_dir=output_dir,
        droplet_count=len(regions),
        mean_diameter_px=float(np.mean(diameters)) if len(diameters) else 0.0,
        std_diameter_px=float(np.std(diameters)) if len(diameters) else 0.0,
        backend=backend,
    )


def mask_nms(predictions: list[Prediction], iou_threshold: float) -> list[Prediction]:
    kept: list[Prediction] = []
    for pred in sorted(predictions, key=lambda item: item.score, reverse=True):
        if not pred.mask.any():
            continue
        if all(mask_iou(pred.mask, other.mask) < iou_threshold for other in kept):
            kept.append(pred)
    return kept


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def masks_to_labels(predictions: list[Prediction], shape: tuple[int, int]) -> np.ndarray:
    label_mask = np.zeros(shape, dtype=np.int32)
    for idx, pred in enumerate(predictions, start=1):
        label_mask[np.logical_and(pred.mask, label_mask == 0)] = idx
    return label_mask


def write_overlay(path: str | Path, image: np.ndarray, regions) -> None:
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for region in regions:
        y, x = region.centroid
        r = max(1, int(equivalent_diameter(region) / 2))
        cv2.circle(overlay, (int(x), int(y)), r, (0, 0, 255), 1)
    cv2.imwrite(str(path), overlay)


def equivalent_diameter(region) -> float:
    if hasattr(region, "equivalent_diameter_area"):
        return float(region.equivalent_diameter_area)
    return float(region.equivalent_diameter)
