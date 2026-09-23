"""Vector stage: turn thresholded terrain into exact, nested pocket regions and disjoint bands.

Correctness properties (enforced here, checked in tests):
  * Cumulative regions are nested: R_1 ⊇ R_2 ⊇ ... ⊇ R_{N-1}.
  * Bands B_k = R_k − R_{k+1} are pairwise disjoint and exactly tile R_1.
  * Holes (a valley inside a ridge, a peak inside a valley) are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon, box
from skimage.measure import find_contours

from topology.layout import BoardLayout
from topology.terrain import Terrain

EMPTY = MultiPolygon()


@dataclass
class Layer:
    index: int  # 0 = top surface (uncut)
    depth_in: float
    elev_low_m: float  # elevation range this layer represents
    elev_high_m: float
    band: MultiPolygon  # area cut to exactly this depth (board inches)
    region: MultiPolygon  # area cut to at least this depth

    @property
    def area_in2(self) -> float:
        return float(self.band.area)


def as_polys(g) -> MultiPolygon:
    """Normalise any shapely result to a valid MultiPolygon, dropping lines/points."""
    if g is None or g.is_empty:
        return EMPTY
    g = shapely.make_valid(g)
    if isinstance(g, Polygon):
        return MultiPolygon([g])
    if isinstance(g, MultiPolygon):
        return g
    parts = [p for p in shapely.get_parts(g) if isinstance(p, (Polygon, MultiPolygon)) and not p.is_empty]
    if not parts:
        return EMPTY
    return as_polys(shapely.union_all(parts))


def region_below(z: np.ndarray, level: float, layout: BoardLayout, min_area_in2: float) -> MultiPolygon:
    """Polygon of {z < level} via marching squares, with sub-pixel smooth edges.

    The array is padded with a value above the level so every contour closes; the
    region is then the even-odd (XOR) combination of the rings, which handles arbitrary
    nesting of islands and holes without relying on ring orientation. Dropping a tiny
    ring removes a tiny island *or* fills a tiny hole – both desirable for a router bit.
    """
    hi = max(float(np.nanmax(z)), level) + 1.0
    padded = np.pad(z, 1, mode="constant", constant_values=hi)
    rings = []
    for c in find_contours(padded, level):
        if len(c) < 4:
            continue
        ring = Polygon(layout.pixels_to_board(c - 1.0))
        if not ring.is_valid:
            ring = as_polys(ring)
        if ring.area >= min_area_in2:
            rings.append(ring)
    return even_odd(rings)


def even_odd(rings: list[Polygon]) -> MultiPolygon:
    """Combine non-crossing closed rings with the even-odd rule.

    A ring's nesting depth is how many other rings contain it: even depth = shell,
    odd depth = hole of its immediate (deepest) container.
    """
    if not rings:
        return EMPTY
    shells = [Polygon(r.exterior) for r in rings]
    arr = np.array(shells, dtype=object)
    inner, outer = shapely.STRtree(arr).query(arr, predicate="within")
    keep = inner != outer
    inner, outer = inner[keep], outer[keep]
    depth = np.bincount(inner, minlength=len(rings))
    parent = np.full(len(rings), -1)
    # For each ring, the container with the greatest depth is its immediate parent.
    order = np.argsort(depth[outer], kind="stable")
    parent[inner[order]] = outer[order]
    holes: dict[int, list] = {}
    for i in np.flatnonzero(depth % 2 == 1):
        holes.setdefault(int(parent[i]), []).append(shells[i].exterior.coords)
    polys = [Polygon(shells[i].exterior.coords, holes.get(int(i), [])) for i in np.flatnonzero(depth % 2 == 0)]
    return as_polys(shapely.union_all(polys))


def build_layers(
    terrain: Terrain,
    th: np.ndarray,
    depths: np.ndarray,
    region_in: MultiPolygon,
    layout: BoardLayout,
    surround_raised: bool,
    min_area_in2: float,
    simplify_in: float,
) -> tuple[list[Layer], MultiPolygon]:
    """Returns the layers (band 0 = top) and the surround pocket (empty in frame mode)."""
    board = box(0, 0, layout.board.width_in, layout.board.height_in)
    region_in = as_polys(region_in)
    surround = as_polys(board.difference(region_in)) if surround_raised else EMPTY

    regions: list[MultiPolygon] = [region_in]  # R_0: everything inside is "at least depth 0"
    for level in th:
        r = region_below(terrain.z, float(level), layout, min_area_in2)
        if simplify_in > 0:
            r = as_polys(r.simplify(simplify_in, preserve_topology=True))
        # Clip to the region and to the previous (shallower) region -> strict nesting.
        r = as_polys(shapely.intersection(r, regions[-1]))
        regions.append(r)
    regions.append(EMPTY)

    elev_edges = np.concatenate([[terrain.zmax], th, [terrain.zmin]])
    layers = []
    for k in range(len(depths)):
        band = as_polys(shapely.difference(regions[k], regions[k + 1])) if not regions[k + 1].is_empty else regions[k]
        layers.append(Layer(
            index=k,
            depth_in=float(depths[k]),
            elev_low_m=float(elev_edges[k + 1]),
            elev_high_m=float(elev_edges[k]),
            band=band,
            region=regions[k],
        ))
    return layers, surround
