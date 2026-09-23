"""Fit a projected region onto the board and define the board raster grid.

Board coordinates are inches with the origin at the top-left corner, x right, y down
(the same convention as SVG), so vector output needs no further transforms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from affine import Affine
from pyproj import CRS
from shapely.geometry import MultiPolygon

from topology.config import BoardSpec


@dataclass(frozen=True)
class BoardLayout:
    board: BoardSpec
    crs: CRS
    center_m: tuple[float, float]
    scale_in_per_m: float
    nx: int
    ny: int

    @property
    def px_w_in(self) -> float:
        return self.board.width_in / self.nx

    @property
    def px_h_in(self) -> float:
        return self.board.height_in / self.ny

    @property
    def shape(self) -> tuple[int, int]:
        return self.ny, self.nx

    @property
    def transform(self) -> Affine:
        """Affine from (col, row) pixel indices to projected metres."""
        s = self.scale_in_per_m
        cx, cy = self.center_m
        return Affine(
            self.px_w_in / s, 0.0, cx - self.board.width_in / (2 * s),
            0.0, -self.px_h_in / s, cy + self.board.height_in / (2 * s),
        )

    @property
    def ground_px_m(self) -> float:
        return min(self.px_w_in, self.px_h_in) / self.scale_in_per_m

    @property
    def map_scale(self) -> float:
        """Representative fraction denominator, e.g. 2_500_000 for 1:2.5M."""
        return 1.0 / (self.scale_in_per_m * 0.0254)

    def to_board(self, geom_m):
        """Projected metres -> board inches (y flipped)."""
        s = self.scale_in_per_m
        cx, cy = self.center_m
        w2, h2 = self.board.width_in / 2, self.board.height_in / 2
        return shapely.transform(geom_m, lambda c: np.column_stack([w2 + s * (c[:, 0] - cx), h2 - s * (c[:, 1] - cy)]))

    def pixels_to_board(self, rc: np.ndarray) -> np.ndarray:
        """(row, col) coordinates in pixel-centre space -> board inches (x, y)."""
        return np.column_stack([(rc[:, 1] + 0.5) * self.px_w_in, (rc[:, 0] + 0.5) * self.px_h_in])


def fit_layout(region_m: MultiPolygon, crs: CRS, board: BoardSpec) -> BoardLayout:
    minx, miny, maxx, maxy = region_m.bounds
    w, h = maxx - minx, maxy - miny
    if w <= 0 or h <= 0:
        raise ValueError("Region has zero extent")
    avail_w = board.width_in - 2 * board.padding_in
    avail_h = board.height_in - 2 * board.padding_in
    scale = min(avail_w / w, avail_h / h)
    nx = max(2, int(round(board.width_in / board.resolution_in)))
    ny = max(2, int(round(board.height_in / board.resolution_in)))
    return BoardLayout(board, crs, ((minx + maxx) / 2, (miny + maxy) / 2), scale, nx, ny)
