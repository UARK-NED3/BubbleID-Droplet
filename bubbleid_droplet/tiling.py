from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Tile:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def iter_tiles(height: int, width: int, tile_size: int, overlap: int) -> Iterator[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must be >= 0 and smaller than tile_size")

    step = tile_size - overlap
    y = 0
    while True:
        x = 0
        y1 = min(y + tile_size, height)
        y0 = max(0, y1 - tile_size)
        while True:
            x1 = min(x + tile_size, width)
            x0 = max(0, x1 - tile_size)
            yield Tile(x0=x0, y0=y0, x1=x1, y1=y1)
            if x1 >= width:
                break
            x += step
        if y1 >= height:
            break
        y += step
