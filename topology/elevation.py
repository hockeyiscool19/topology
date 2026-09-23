"""Elevation data: AWS Terrain Tiles (Terrarium encoding), warped straight onto the board grid.

Fetching at the zoom that matches the board's ground resolution (instead of a fixed 30 m) keeps
work proportional to the output size, so a large state costs about the same as a small one.
"""

from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

import numpy as np
from affine import Affine
from PIL import Image
from rasterio.crs import CRS as RioCRS
from rasterio.warp import Resampling, reproject, transform_bounds

from topology.cache import cache_dir, fetch_bytes
from topology.layout import BoardLayout

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE = 256
MAX_ZOOM = 15
MAX_TILES = 400
WEB_MERC_HALF = 20037508.342789244
MAX_LAT = 85.05112878


class ElevationSource(Protocol):
    def sample(self, layout: BoardLayout) -> np.ndarray:
        """Return elevation in metres on the layout grid (ny, nx), NaN where unknown."""
        ...


def _lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    lat = max(-MAX_LAT, min(MAX_LAT, lat))
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def choose_zoom(bounds: tuple[float, float, float, float], ground_px_m: float) -> int:
    """Coarsest zoom whose pixels are at least as fine as the board pixels, capped by tile count."""
    west, south, east, north = bounds
    lat = max(abs(south), abs(north)) if south * north > 0 else 0.0
    circ = 2 * WEB_MERC_HALF * math.cos(math.radians(min(lat, MAX_LAT)))
    z = int(math.ceil(math.log2(max(circ / (TILE * ground_px_m), 1.0))))
    z = max(0, min(MAX_ZOOM, z))
    while z > 0 and _tile_count(bounds, z) > MAX_TILES:
        z -= 1
    return z


def _tile_range(bounds, z):
    west, south, east, north = bounds
    x0, y0 = _lonlat_to_tile(west, north, z)
    x1, y1 = _lonlat_to_tile(east, south, z)
    n = 2**z
    return (max(0, int(math.floor(x0))), min(n - 1, int(math.floor(x1))),
            max(0, int(math.floor(y0))), min(n - 1, int(math.floor(y1))))


def _tile_count(bounds, z) -> int:
    tx0, tx1, ty0, ty1 = _tile_range(bounds, z)
    return (tx1 - tx0 + 1) * (ty1 - ty0 + 1)


def _decode_terrarium(png: bytes) -> np.ndarray:
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


class TerrariumSource:
    def __init__(self, url: str = TERRARIUM_URL, workers: int = 16):
        self.url = url
        self.workers = workers

    def _tile(self, z: int, x: int, y: int) -> np.ndarray:
        path = cache_dir("terrarium", str(z), str(x)) / f"{y}.png"
        if path.exists() and path.stat().st_size > 0:
            data = path.read_bytes()
        else:
            data = fetch_bytes(self.url.format(z=z, x=x, y=y))
            tmp = path.with_suffix(".part")
            tmp.write_bytes(data)
            tmp.replace(path)
        return _decode_terrarium(data)

    def mosaic(self, bounds, z: int) -> tuple[np.ndarray, Affine]:
        tx0, tx1, ty0, ty1 = _tile_range(bounds, z)
        keys = [(x, y) for y in range(ty0, ty1 + 1) for x in range(tx0, tx1 + 1)]
        with ThreadPoolExecutor(self.workers) as ex:
            tiles = list(ex.map(lambda k: self._tile(z, *k), keys))
        out = np.empty(((ty1 - ty0 + 1) * TILE, (tx1 - tx0 + 1) * TILE), dtype=np.float32)
        for (x, y), t in zip(keys, tiles):
            r, c = (y - ty0) * TILE, (x - tx0) * TILE
            out[r:r + TILE, c:c + TILE] = t
        res = 2 * WEB_MERC_HALF / (TILE * 2**z)
        transform = Affine(res, 0, -WEB_MERC_HALF + tx0 * TILE * res, 0, -res, WEB_MERC_HALF - ty0 * TILE * res)
        return out, transform

    def sample(self, layout: BoardLayout) -> np.ndarray:
        dst_crs = RioCRS.from_wkt(layout.crs.to_wkt())
        ny, nx = layout.shape
        t = layout.transform
        west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", *_grid_bounds(t, nx, ny), densify_pts=51)
        # Across the antimeridian transform_bounds returns east < west: fetch each side separately
        # instead of the whole globe (which would force a uselessly coarse zoom).
        spans = [(west, east)] if east >= west else [(west, 180.0), (-180.0, east)]
        dst = np.full((ny, nx), np.nan, dtype=np.float32)
        for w, e in spans:
            bounds = (w, south, e, north)
            src, src_t = self.mosaic(bounds, choose_zoom(bounds, layout.ground_px_m))
            part = np.full((ny, nx), np.nan, dtype=np.float32)
            reproject(
                src, part,
                src_transform=src_t, src_crs="EPSG:3857",
                dst_transform=t, dst_crs=dst_crs,
                resampling=Resampling.bilinear, dst_nodata=np.nan,
            )
            dst = np.where(np.isnan(dst), part, dst)
        return dst


def _grid_bounds(t: Affine, nx: int, ny: int) -> tuple[float, float, float, float]:
    x0, y0 = t.c, t.f
    x1, y1 = t.c + t.a * nx, t.f + t.e * ny
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
