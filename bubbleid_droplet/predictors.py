from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Prediction:
    mask: np.ndarray
    score: float
    bbox: tuple[int, int, int, int]


class ClassicalDropletPredictor:
    """Small dependency-free smoke-test backend, not a replacement for BubbleID."""

    def __init__(self, min_area: int = 8, max_area_fraction: float = 0.30) -> None:
        self.min_area = min_area
        self.max_area_fraction = max_area_fraction

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

        dark = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        bright = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        mask = dark if np.count_nonzero(dark) < np.count_nonzero(bright) else bright
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        predictions: list[Prediction] = []
        max_area = image.shape[0] * image.shape[1] * self.max_area_fraction
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_area or area > max_area:
                continue
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            predictions.append(
                Prediction(mask=labels == label, score=0.5, bbox=(x, y, x + w, y + h))
            )
        return predictions


class Detectron2BubbleIDPredictor:
    def __init__(
        self,
        weights: str | Path,
        score_threshold: float = 0.10,
        device: str = "cuda",
        min_size_test: int = 800,
        max_size_test: int = 1333,
        detections_per_image: int = 3000,
        num_classes: int = 1,
    ) -> None:
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise ImportError(
                "Detectron2 is required for the BubbleID model backend. "
                "Install the project in a Python environment with torch and detectron2."
            ) from exc

        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.WEIGHTS = str(weights)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.DEVICE = device
        cfg.INPUT.MIN_SIZE_TEST = min_size_test
        cfg.INPUT.MAX_SIZE_TEST = max_size_test
        cfg.TEST.DETECTIONS_PER_IMAGE = detections_per_image
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = max(detections_per_image, 1000)
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = max(detections_per_image * 2, 6000)
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, image: np.ndarray) -> list[Prediction]:
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")
        if not instances.has("pred_masks"):
            return []
        masks = instances.pred_masks.numpy().astype(bool)
        scores = instances.scores.numpy() if instances.has("scores") else np.ones(len(masks))
        boxes = instances.pred_boxes.tensor.numpy().astype(int)

        predictions: list[Prediction] = []
        for mask, score, box in zip(masks, scores, boxes):
            predictions.append(Prediction(mask=mask, score=float(score), bbox=tuple(box.tolist())))
        return predictions
