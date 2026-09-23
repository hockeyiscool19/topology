"""Job configuration. All physical board dimensions are in inches."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

Surround = Literal["raised", "frame"]
Curve = Literal["linear", "equal-area"]


@dataclass(frozen=True)
class BoardSpec:
    width_in: float = 11.0
    height_in: float = 8.0
    thickness_in: float = 0.75
    padding_in: float = 0.25
    # Material that is never cut, so the board keeps its integrity.
    floor_in: float = 0.25
    # Raster resolution in board inches per pixel. 0.01" is far finer than any router bit.
    resolution_in: float = 0.01

    def validate(self) -> None:
        if self.width_in <= 0 or self.height_in <= 0:
            raise ValueError("Board width and height must be positive")
        if self.thickness_in <= 0:
            raise ValueError("Board thickness must be positive")
        if self.padding_in < 0 or 2 * self.padding_in >= min(self.width_in, self.height_in):
            raise ValueError("Padding must be >= 0 and leave room for the design")
        if not 0 <= self.floor_in < self.thickness_in:
            raise ValueError("Floor must be >= 0 and thinner than the board")
        if not 0.001 <= self.resolution_in <= 0.25:
            raise ValueError("Resolution must be between 0.001 and 0.25 inches")
        if self.pixel_count > 12_000_000:
            raise ValueError("Board raster too large; increase resolution_in")

    @property
    def max_cut_in(self) -> float:
        return self.thickness_in - self.floor_in

    @property
    def pixel_count(self) -> int:
        return int(self.width_in / self.resolution_in) * int(self.height_in / self.resolution_in)


@dataclass(frozen=True)
class LayerSpec:
    # Number of distinct heights, including the uncut top surface.
    count: int = 8
    # Deepest pocket depth; defaults to the full cuttable depth of the board.
    max_depth_in: Optional[float] = None
    # Gaussian smoothing of the terrain, in board inches (physical, resolution independent).
    smoothing_in: float = 0.04
    # linear: equal elevation bands. equal-area: each layer covers a similar share of the map.
    curve: Curve = "linear"
    # >1 exaggerates peaks (more layers spent on high ground), <1 exaggerates lowlands.
    exaggeration: float = 1.0
    # Islands/holes smaller than this (square inches) are dropped; keeps the bit out of specks.
    min_feature_in2: float = 0.005
    # raised: area outside the boundary is cut to max depth so the region stands proud.
    # frame: area outside the boundary is left at the top surface.
    surround: Surround = "raised"
    # Vector simplification tolerance, board inches.
    simplify_in: float = 0.002

    def validate(self, board: BoardSpec) -> None:
        if not 2 <= self.count <= 64:
            raise ValueError("Layer count must be between 2 and 64")
        if self.max_depth_in is not None and not 0 < self.max_depth_in <= board.max_cut_in + 1e-9:
            raise ValueError(f"Max depth must be in (0, {board.max_cut_in:.3f}] (thickness - floor)")
        if self.smoothing_in < 0:
            raise ValueError("Smoothing must be >= 0")
        if self.curve not in ("linear", "equal-area"):
            raise ValueError("curve must be 'linear' or 'equal-area'")
        if not 0.1 <= self.exaggeration <= 10:
            raise ValueError("Exaggeration must be between 0.1 and 10")
        if self.surround not in ("raised", "frame"):
            raise ValueError("surround must be 'raised' or 'frame'")
        if self.min_feature_in2 < 0 or self.simplify_in < 0:
            raise ValueError("min_feature_in2 and simplify_in must be >= 0")

    def depth_in(self, board: BoardSpec) -> float:
        return board.max_cut_in if self.max_depth_in is None else float(self.max_depth_in)


@dataclass(frozen=True)
class JobSpec:
    # Either a US state (name or USPS code) or a GeoJSON geometry in WGS84.
    state: Optional[str] = None
    geojson: Optional[dict[str, Any]] = None
    name: Optional[str] = None
    board: BoardSpec = field(default_factory=BoardSpec)
    layers: LayerSpec = field(default_factory=LayerSpec)
    # Keep only the largest polygon of the region (drops small islands).
    largest_only: bool = False

    def validate(self) -> None:
        if (self.state is None) == (self.geojson is None):
            raise ValueError("Provide exactly one of 'state' or 'geojson'")
        self.board.validate()
        self.layers.validate(self.board)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JobSpec":
        d = dict(d)
        board = BoardSpec(**(d.pop("board", None) or {}))
        layers = LayerSpec(**(d.pop("layers", None) or {}))
        return cls(board=board, layers=layers, **d)
