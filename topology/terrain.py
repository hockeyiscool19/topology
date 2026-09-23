"""Raster stage: masking, smoothing and elevation thresholds on the board grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely.geometry import MultiPolygon

from topology.config import LayerSpec
from topology.layout import BoardLayout


@dataclass
class Terrain:
    # Smoothed elevation (m) on the board grid; outside the region it is extended from the
    # nearest inside pixel so contours don't bend toward the boundary.
    z: np.ndarray
    mask: np.ndarray  # True inside the region
    zmin: float
    zmax: float


def region_mask(region_m: MultiPolygon, layout: BoardLayout) -> np.ndarray:
    return rasterize(
        [(region_m, 1)], out_shape=layout.shape, transform=layout.transform, fill=0, dtype="uint8"
    ).astype(bool)


def prepare(z_raw: np.ndarray, mask: np.ndarray, layout: BoardLayout, spec: LayerSpec) -> Terrain:
    valid = mask & np.isfinite(z_raw)
    if valid.sum() < 4:
        raise ValueError("No elevation data inside the region (is it too small for the board resolution?)")

    z = np.where(valid, z_raw, 0.0).astype(np.float64)
    sigma = spec.smoothing_in / min(layout.px_w_in, layout.px_h_in)
    if sigma > 0:
        # Normalised convolution: smooth only over valid pixels so the boundary doesn't
        # drag elevations toward zero/nodata.
        w = gaussian_filter(valid.astype(np.float64), sigma, mode="nearest")
        num = gaussian_filter(z, sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            z = np.where(w > 1e-6, num / w, 0.0)
        has = w > 1e-6
    else:
        has = valid

    zin = z[valid]
    # Robust floor: coastline pixels pick up ocean bathymetry, which would otherwise stretch
    # the range and squash every land layer. Peaks keep their true maximum.
    zmin, zmax = float(np.percentile(zin, 0.2)), float(zin.max())
    if zmax <= zmin:
        zmin = float(zin.min())

    # Extend values into unknown pixels from the nearest known pixel.
    idx = distance_transform_edt(~has, return_distances=False, return_indices=True)
    z = z[tuple(idx)]
    return Terrain(z=z, mask=mask, zmin=zmin, zmax=zmax)


def thresholds(t: Terrain, spec: LayerSpec) -> np.ndarray:
    """Descending elevations T_1 > ... > T_{N-1}; band k holds T_{k+1} <= z < T_k."""
    n = spec.count
    q = (1.0 - np.arange(1, n) / n) ** (1.0 / spec.exaggeration)
    if t.zmax <= t.zmin:
        return np.full(n - 1, t.zmin, dtype=float)
    if spec.curve == "equal-area":
        return np.quantile(t.z[t.mask], q)
    return t.zmin + q * (t.zmax - t.zmin)


def band_depths(spec: LayerSpec, max_depth_in: float) -> tuple[np.ndarray, float | None]:
    """Depth per band (band 0 = uncut top) and the depth of the area outside the region.

    In 'raised' mode the surround takes the deepest level so the lowest land still stands
    one step proud of it and the coastline stays visible.
    """
    n = spec.count
    if spec.surround == "raised":
        step = max_depth_in / n
        return np.arange(n) * step, max_depth_in
    step = max_depth_in / (n - 1)
    return np.arange(n) * step, None


def smooth_depth(t: Terrain, th: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """Continuous depth field matching the stepped layers at each threshold (for smooth STL)."""
    # Knots: zmax -> depth 0, T_k -> depth of band k, zmin -> deepest band. np.interp needs ascending x.
    xs = np.concatenate([[t.zmin], th[::-1], [t.zmax]])
    ys = np.concatenate([[depths[-1]], depths[1:][::-1], [0.0]])
    xs, keep = np.unique(xs, return_index=True)
    return np.interp(t.z, xs, ys[keep])
