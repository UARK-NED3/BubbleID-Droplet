import numpy as np
import unittest

from bubbleid_droplet.pipeline import Prediction, mask_nms
from bubbleid_droplet.tiling import iter_tiles


class TilingTests(unittest.TestCase):
    def test_iter_tiles_covers_edges(self):
        tiles = list(iter_tiles(height=100, width=130, tile_size=64, overlap=16))
        self.assertEqual(tiles[0].x0, 0)
        self.assertEqual(tiles[0].y0, 0)
        self.assertEqual(tiles[-1].x1, 130)
        self.assertEqual(tiles[-1].y1, 100)

    def test_mask_nms_removes_duplicate_mask(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[2:8, 2:8] = True
        duplicate = mask.copy()
        separate = np.zeros((16, 16), dtype=bool)
        separate[10:14, 10:14] = True

        kept = mask_nms(
            [
                Prediction(mask=mask, score=0.9, bbox=(2, 2, 8, 8)),
                Prediction(mask=duplicate, score=0.8, bbox=(2, 2, 8, 8)),
                Prediction(mask=separate, score=0.7, bbox=(10, 10, 14, 14)),
            ],
            iou_threshold=0.3,
        )

        self.assertEqual(len(kept), 2)


if __name__ == "__main__":
    unittest.main()
