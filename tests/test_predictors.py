import tempfile
import unittest
from pathlib import Path

import torch

from bubbleid_droplet.predictors import infer_detectron2_num_classes


class PredictorTests(unittest.TestCase):
    def test_infer_num_classes_from_mask_head(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save(
                {"model": {"roi_heads.mask_head.predictor.weight": torch.zeros(2, 256, 1, 1)}},
                path,
            )

            self.assertEqual(infer_detectron2_num_classes(path), 2)

    def test_infer_num_classes_from_box_head_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pth"
            torch.save(
                {"model": {"roi_heads.box_predictor.cls_score.weight": torch.zeros(3, 1024)}},
                path,
            )

            self.assertEqual(infer_detectron2_num_classes(path), 2)


if __name__ == "__main__":
    unittest.main()
