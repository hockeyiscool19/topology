import math
import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import py3dep
import rioxarray  # noqa: F401
import rioxarray as rxr
from py3dep.exceptions import ServiceUnavailableError
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
from shapely import affinity
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union


CB_STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_20m.zip"


@dataclass
class MaterialSpec:
    target_width_in: float
    target_height_in: float
    padding_in: float
    thickness_in: float


@dataclass
class ContourSpec:
    resolution_m: int
    interval_m: float
    smooth_sigma_px: float


def _normalize_state_query(q: str) -> str:
    return " ".join(q.strip().upper().split())


def _load_states_gdf() -> gpd.GeoDataFrame:
    return gpd.read_file(CB_STATES_URL)


def _select_state(gdf: gpd.GeoDataFrame, query: str) -> gpd.GeoSeries:
    q = _normalize_state_query(query)
    if "STUSPS" in gdf.columns:
        m = gdf["STUSPS"].astype(str).str.upper() == q
        if m.any():
            return gdf.loc[m].iloc[0]

    if "NAME" in gdf.columns:
        m = gdf["NAME"].astype(str).str.upper() == q
        if m.any():
            return gdf.loc[m].iloc[0]

    raise ValueError(f"State not found for query: {query!r}")


def _largest_polygon(geom) -> Polygon:
    if geom is None:
        raise ValueError("Empty geometry")

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
        if not polys:
            raise ValueError("Empty MultiPolygon")
        return max(polys, key=lambda p: p.area)

    merged = unary_union(geom)
    if isinstance(merged, Polygon):
        return merged
    if isinstance(merged, MultiPolygon):
        return max(list(merged.geoms), key=lambda p: p.area)

    raise ValueError(f"Unsupported geometry type: {type(geom)}")


def _ensure_closed_ring(coords: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not coords:
        return []
    out = list(coords)
    if out[0] != out[-1]:
        out.append(out[0])
    return out


def _chaikin_smooth_closed(coords: Sequence[Tuple[float, float]], iterations: int) -> List[Tuple[float, float]]:
    pts = _ensure_closed_ring(coords)
    if iterations <= 0 or len(pts) < 4:
        return pts

    for _ in range(iterations):
        new_pts: List[Tuple[float, float]] = []
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            qx, qy = 0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1
            rx, ry = 0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1
            if i == 0:
                new_pts.append((qx, qy))
            else:
                new_pts.append((qx, qy))
            new_pts.append((rx, ry))
        if new_pts and new_pts[0] != new_pts[-1]:
            new_pts.append(new_pts[0])
        pts = new_pts

    return pts


def _coords_to_svg_path(coords: Sequence[Tuple[float, float]]) -> str:
    pts = _ensure_closed_ring(coords)
    if len(pts) < 4:
        raise ValueError("Not enough points to form a closed path")

    parts = [f"M {pts[0][0]:.3f} {pts[0][1]:.3f}"]
    for x, y in pts[1:]:
        parts.append(f"L {x:.3f} {y:.3f}")
    parts.append("Z")
    return " ".join(parts)


def _write_svg(paths: Sequence[str], svg_w_m: float, svg_h_m: float, output_svg_path: str) -> str:
    svg_w_in = svg_w_m / 0.0254
    svg_h_in = svg_h_m / 0.0254

    os.makedirs(os.path.dirname(output_svg_path) or ".", exist_ok=True)

    svg = (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{svg_w_in:.4f}in\" height=\"{svg_h_in:.4f}in\" "
        f"viewBox=\"0 0 {svg_w_m:.6f} {svg_h_m:.6f}\">"
        + "".join(paths)
        + "</svg>"
    )

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return output_svg_path


def _depth_from_elevation_m(elev_m: float, zmin: float, zmax: float, max_depth_in: float) -> float:
    if zmax <= zmin:
        return 0.0
    t = (zmax - float(elev_m)) / (zmax - zmin)
    return max(0.0, min(1.0, t)) * float(max_depth_in)


def _quantize_depth_in(depth_in: float, stepdown_in: float) -> float:
    if stepdown_in <= 0:
        return float(depth_in)
    return math.ceil(float(depth_in) / float(stepdown_in)) * float(stepdown_in)


def _elevation_from_depth_in(depth_in: float, zmin: float, zmax: float, max_depth_in: float) -> float:
    if max_depth_in <= 0 or zmax <= zmin:
        return float(zmax)
    t = max(0.0, min(1.0, float(depth_in) / float(max_depth_in)))
    return float(zmax) - t * (float(zmax) - float(zmin))


def _polygon_to_svg_path(p: Polygon) -> str:
    d_parts: List[str] = []
    d_parts.append(_coords_to_svg_path(list(p.exterior.coords)))
    for ring in p.interiors:
        d_parts.append(_coords_to_svg_path(list(ring.coords)))
    return " ".join(d_parts)


def _geom_to_svg_paths_filled(geom) -> List[str]:
    if geom is None:
        return []

    if isinstance(geom, Polygon):
        d = _polygon_to_svg_path(geom)
        return [
            f"<path d=\"{d}\" fill=\"#000000\" stroke=\"none\" fill-rule=\"evenodd\" />"
        ]

    if isinstance(geom, MultiPolygon):
        out: List[str] = []
        for p in geom.geoms:
            out.extend(_geom_to_svg_paths_filled(p))
        return out

    return []


def _pocket_region_below_threshold(
    z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    elev_threshold_m: float,
) -> Optional[Polygon]:
    if z.ndim != 2:
        raise ValueError("Unexpected DEM array shape")

    valid = np.isfinite(z)
    if not np.any(valid):
        return None

    mask = np.zeros_like(z, dtype=float)
    mask[valid & (z <= float(elev_threshold_m))] = 1.0

    contours = find_contours(mask, level=0.5)
    polys: List[Polygon] = []
    for c in contours:
        if c.shape[0] < 8:
            continue
        rr = np.clip(np.round(c[:, 0]).astype(int), 0, len(ys) - 1)
        cc = np.clip(np.round(c[:, 1]).astype(int), 0, len(xs) - 1)
        coords = list(zip(xs[cc], ys[rr]))
        if len(coords) < 4:
            continue
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            continue
        polys.append(p)

    if not polys:
        return None

    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        return merged
    if isinstance(merged, MultiPolygon):
        return unary_union(list(merged.geoms))
    return None


def _pocket_region_in_band(
    z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    elev_low_m: float,
    elev_high_m: float,
) -> Optional[Polygon]:
    if z.ndim != 2:
        raise ValueError("Unexpected DEM array shape")

    valid = np.isfinite(z)
    if not np.any(valid):
        return None

    lo = float(elev_low_m)
    hi = float(elev_high_m)
    if hi <= lo:
        return None

    mask = np.zeros_like(z, dtype=float)
    mask[valid & (z > lo) & (z <= hi)] = 1.0

    contours = find_contours(mask, level=0.5)
    polys: List[Polygon] = []
    for c in contours:
        if c.shape[0] < 8:
            continue
        rr = np.clip(np.round(c[:, 0]).astype(int), 0, len(ys) - 1)
        cc = np.clip(np.round(c[:, 1]).astype(int), 0, len(xs) - 1)
        coords = list(zip(xs[cc], ys[rr]))
        if len(coords) < 4:
            continue
        p = Polygon(coords)
        if not p.is_valid:
            p = p.buffer(0)
        if p.is_empty or p.area <= 0:
            continue
        polys.append(p)

    if not polys:
        return None

    merged = unary_union(polys)
    if isinstance(merged, Polygon):
        return merged
    if isinstance(merged, MultiPolygon):
        return unary_union(list(merged.geoms))
    return None


def _aggregate_dem_max_by_cell_m(dem_clip, cell_m: float):
    if cell_m <= 0:
        return dem_clip

    try:
        resx, resy = dem_clip.rio.resolution()
        resx = abs(float(resx))
        resy = abs(float(resy))
    except Exception:
        xs = np.asarray(dem_clip["x"].values, dtype=float)
        ys = np.asarray(dem_clip["y"].values, dtype=float)
        resx = abs(float(xs[1] - xs[0])) if xs.size >= 2 else float(cell_m)
        resy = abs(float(ys[1] - ys[0])) if ys.size >= 2 else float(cell_m)

    fx = max(1, int(round(float(cell_m) / resx)))
    fy = max(1, int(round(float(cell_m) / resy)))
    if fx == 1 and fy == 1:
        return dem_clip

    return dem_clip.coarsen(x=fx, y=fy, boundary="trim").max(skipna=True)


def _to_projected_meters(gdf: gpd.GeoDataFrame, state_row: gpd.GeoSeries) -> Polygon:
    geom = _largest_polygon(state_row.geometry)
    state_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=gdf.crs)
    epsg = 5070
    if "STUSPS" in state_row and str(state_row["STUSPS"]).upper() == "AK":
        epsg = 3338
    elif "STUSPS" in state_row and str(state_row["STUSPS"]).upper() == "HI":
        epsg = 3857
    proj = state_gdf.to_crs(epsg=epsg)
    return _largest_polygon(proj.iloc[0].geometry)


def _to_wgs84(gdf: gpd.GeoDataFrame, state_row: gpd.GeoSeries) -> Polygon:
    geom = _largest_polygon(state_row.geometry)
    state_gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs=gdf.crs)
    wgs = state_gdf.to_crs(epsg=4326)
    return _largest_polygon(wgs.iloc[0].geometry)


def _scale_to_material(poly_m: Polygon, spec: MaterialSpec) -> Tuple[Polygon, float, float, float, float, float]:
    minx, miny, maxx, maxy = poly_m.bounds
    width_m = maxx - minx
    height_m = maxy - miny
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Invalid bounds for state geometry")

    pad_m = spec.padding_in * 0.0254
    avail_w_m = max(spec.target_width_in * 0.0254 - 2.0 * pad_m, 1e-9)
    avail_h_m = max(spec.target_height_in * 0.0254 - 2.0 * pad_m, 1e-9)

    scale_w = avail_w_m / width_m
    scale_h = avail_h_m / height_m
    scale = min(scale_w, scale_h)
    scaled = affinity.scale(poly_m, xfact=scale, yfact=scale, origin=(minx, miny))

    minx2, miny2, maxx2, maxy2 = scaled.bounds
    scaled = affinity.translate(scaled, xoff=-(minx2 - pad_m), yoff=-(miny2 - pad_m))
    minx3, miny3, maxx3, maxy3 = scaled.bounds

    svg_w_m = (maxx3 - minx3) + pad_m
    svg_h_m = (maxy3 - miny3) + pad_m

    target_width_m = spec.target_width_in * 0.0254
    return scaled, scale, svg_w_m, svg_h_m, pad_m, target_width_m


def _flip_y_for_svg(poly_m: Polygon, svg_h_m: float) -> Polygon:
    flipped = affinity.scale(poly_m, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
    return affinity.translate(flipped, xoff=0.0, yoff=svg_h_m)


def _flip_y_for_svg_lines(lines: Sequence[LineString], svg_h_m: float) -> List[LineString]:
    out: List[LineString] = []
    for ln in lines:
        flipped = affinity.scale(ln, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
        out.append(affinity.translate(flipped, xoff=0.0, yoff=svg_h_m))
    return out


def _scale_lines_to_material(
    lines_m: Sequence[LineString], spec: MaterialSpec
) -> Tuple[List[LineString], float, float, float]:
    if not lines_m:
        raise ValueError("No contour lines generated")

    minx = min(ln.bounds[0] for ln in lines_m)
    miny = min(ln.bounds[1] for ln in lines_m)
    maxx = max(ln.bounds[2] for ln in lines_m)
    maxy = max(ln.bounds[3] for ln in lines_m)

    width_m = maxx - minx
    height_m = maxy - miny
    if width_m <= 0 or height_m <= 0:
        raise ValueError("Invalid bounds for contour geometry")

    pad_m = spec.padding_in * 0.0254
    avail_w_m = max(spec.target_width_in * 0.0254 - 2.0 * pad_m, 1e-9)
    avail_h_m = max(spec.target_height_in * 0.0254 - 2.0 * pad_m, 1e-9)

    scale_w = avail_w_m / width_m
    scale_h = avail_h_m / height_m
    scale = min(scale_w, scale_h)

    scaled: List[LineString] = []
    for ln in lines_m:
        scaled.append(affinity.scale(ln, xfact=scale, yfact=scale, origin=(minx, miny)))

    minx2 = min(ln.bounds[0] for ln in scaled)
    miny2 = min(ln.bounds[1] for ln in scaled)
    maxx2 = max(ln.bounds[2] for ln in scaled)
    maxy2 = max(ln.bounds[3] for ln in scaled)

    translated: List[LineString] = []
    for ln in scaled:
        translated.append(affinity.translate(ln, xoff=-(minx2 - pad_m), yoff=-(miny2 - pad_m)))

    minx3 = min(ln.bounds[0] for ln in translated)
    miny3 = min(ln.bounds[1] for ln in translated)
    maxx3 = max(ln.bounds[2] for ln in translated)
    maxy3 = max(ln.bounds[3] for ln in translated)

    svg_w_m = (maxx3 - minx3) + pad_m
    svg_h_m = (maxy3 - miny3) + pad_m

    return translated, scale, svg_w_m, svg_h_m


def _line_to_svg_path_open(coords: Sequence[Tuple[float, float]]) -> str:
    if len(coords) < 2:
        raise ValueError("Not enough points to form a path")

    parts = [f"M {coords[0][0]:.3f} {coords[0][1]:.3f}"]
    for x, y in coords[1:]:
        parts.append(f"L {x:.3f} {y:.3f}")
    return " ".join(parts)


def export_state_contours_svg(
    state_query: str,
    target_width_in: float,
    target_height_in: float,
    output_svg_path: str,
    padding_in: float = 0.25,
    thickness_in: float = 0.75,
    max_depth_in: Optional[float] = None,
    stepdown_in: float = 0.115,
    export_per_layer: bool = False,
    include_outline_final_cut: bool = True,
    low_res_1km_max: bool = False,
    low_res_cell_m: float = 1000.0,
    dem_cache_tif: Optional[str] = None,
    dem_resolution_m: int = 30,
    contour_interval_m: float = 100.0,
    smooth_sigma_px: float = 1.5,
    max_contours: int = 4000,
    output_kind: str = "lines",
) -> str:
    gdf = _load_states_gdf()
    row = _select_state(gdf, state_query)

    state_wgs84 = _to_wgs84(gdf, row)
    poly_target = _to_projected_meters(gdf, row)

    if dem_cache_tif and os.path.exists(dem_cache_tif):
        dem_clip = rxr.open_rasterio(dem_cache_tif).squeeze(drop=True)
        if not hasattr(dem_clip, "rio") or dem_clip.rio.crs is None:
            raise ValueError("Cached DEM missing CRS information")
    else:
        last_err: Optional[Exception] = None
        dem = None
        for attempt in range(4):
            try:
                dem = py3dep.get_map("DEM", state_wgs84, resolution=dem_resolution_m, geo_crs=4326, crs=3857)
                last_err = None
                break
            except ServiceUnavailableError as e:
                last_err = e
                dem = None
                break
            except Exception as e:
                last_err = e
                time_s = min(60.0, 2.0**attempt)
                time.sleep(time_s)

        if dem is None:
            try:
                dem = py3dep.get_dem(state_wgs84, dem_resolution_m)
                last_err = None
            except Exception as e:
                last_err = e

        if last_err is not None or dem is None:
            cache_hint = f" You can also set DEM cache GeoTIFF path and rerun to avoid repeated downloads." if dem_cache_tif else ""
            raise RuntimeError(
                "Failed to retrieve DEM from USGS 3DEP services." + cache_hint + f" Last error: {last_err}"
            )

        dem_5070 = dem.rio.reproject(5070)
        clip_geom = gpd.GeoDataFrame({"geometry": [poly_target]}, crs=5070)
        dem_clip = dem_5070.rio.clip(clip_geom.geometry, clip_geom.crs, drop=True)

        if dem_cache_tif:
            os.makedirs(os.path.dirname(dem_cache_tif) or ".", exist_ok=True)
            dem_clip.rio.to_raster(dem_cache_tif)

    if low_res_1km_max:
        dem_clip = _aggregate_dem_max_by_cell_m(dem_clip, float(low_res_cell_m))

    z = dem_clip.values
    if z.ndim != 2:
        z = np.asarray(z).squeeze()
    if z.ndim != 2:
        raise ValueError("Unexpected DEM array shape")

    z = np.where(np.isfinite(z), z, np.nan)
    nodata = None
    try:
        nodata = dem_clip.rio.nodata
    except Exception:
        nodata = None
    if nodata is None:
        nodata = dem_clip.attrs.get("_FillValue")
    if nodata is not None:
        try:
            nodata_f = float(nodata)
        except Exception:
            nodata_f = None
        if nodata_f is not None and np.isfinite(nodata_f):
            z = np.where(np.isclose(z, nodata_f), np.nan, z)
    zmin = np.nanmin(z)
    zmax = np.nanmax(z)
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError("DEM has invalid elevation range")

    if max_depth_in is None:
        max_depth_in = float(thickness_in)
    max_depth_in = max(0.0, min(float(max_depth_in), float(thickness_in)))
    stepdown_in = max(0.0, float(stepdown_in))

    uncut_thickness_in = 0.25
    max_cut_in = max(0.0, float(thickness_in) - float(uncut_thickness_in))
    max_cut_in = min(max_cut_in, float(max_depth_in))

    z0 = np.nan_to_num(z, nan=float(zmin))
    if smooth_sigma_px > 0:
        z0 = gaussian_filter(z0, sigma=float(smooth_sigma_px))

    start = math.ceil(zmin / contour_interval_m) * contour_interval_m
    levels = np.arange(start, zmax, contour_interval_m, dtype=float)
    if levels.size == 0:
        levels = np.array([(zmin + zmax) / 2.0], dtype=float)

    xs = np.asarray(dem_clip["x"].values, dtype=float)
    ys = np.asarray(dem_clip["y"].values, dtype=float)
    raw_lines: List[Tuple[LineString, float]] = []

    for level in levels:
        contours = find_contours(z0, level=float(level))
        for c in contours:
            if c.shape[0] < 8:
                continue
            rr = np.clip(np.round(c[:, 0]).astype(int), 0, len(ys) - 1)
            cc = np.clip(np.round(c[:, 1]).astype(int), 0, len(xs) - 1)
            coords = list(zip(xs[cc], ys[rr]))
            ln = LineString(coords)
            if ln.length <= 0:
                continue
            raw_lines.append((ln, float(level)))
            if len(raw_lines) >= max_contours:
                break
        if len(raw_lines) >= max_contours:
            break

    spec = MaterialSpec(
        target_width_in=target_width_in,
        target_height_in=target_height_in,
        padding_in=padding_in,
        thickness_in=thickness_in,
    )

    outline_scaled, scale, svg_w_m, svg_h_m, pad_m, _ = _scale_to_material(poly_target, spec)

    minx, miny, maxx, maxy = poly_target.bounds
    poly_scaled_only = affinity.scale(poly_target, xfact=scale, yfact=scale, origin=(minx, miny))
    minx2, miny2, _, _ = poly_scaled_only.bounds
    xoff = -(minx2 - pad_m)
    yoff = -(miny2 - pad_m)

    only_lines = [ln for ln, _ in raw_lines]
    scaled_lines: List[LineString] = []
    for ln in only_lines:
        ln2 = affinity.scale(ln, xfact=scale, yfact=scale, origin=(minx, miny))
        ln2 = affinity.translate(ln2, xoff=xoff, yoff=yoff)
        scaled_lines.append(ln2)

    flipped_lines = _flip_y_for_svg_lines(scaled_lines, svg_h_m)
    outline_flipped = _flip_y_for_svg(outline_scaled, svg_h_m)

    output_kind = (output_kind or "lines").strip().lower()

    if output_kind not in {"lines", "pockets"}:
        raise ValueError("output_kind must be 'lines' or 'pockets'")

    if output_kind == "pockets" and not export_per_layer:
        raise ValueError("Pocket output requires export_per_layer=True")

    if not export_per_layer:
        path_elems: List[str] = []
        for (orig_ln, elev_m), ln in zip(raw_lines, flipped_lines):
            depth_in = _depth_from_elevation_m(elev_m, float(zmin), float(zmax), float(max_depth_in))
            depth_q = _quantize_depth_in(depth_in, float(stepdown_in))
            d = _line_to_svg_path_open(list(ln.coords))
            path_elems.append(
                f"<path d=\"{d}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.001\" "
                f"data-elev-m=\"{elev_m:.3f}\" data-depth-in=\"{depth_q:.4f}\" />"
            )
        if include_outline_final_cut:
            outline_d = _coords_to_svg_path(list(outline_flipped.exterior.coords))
            final_depth_q = _quantize_depth_in(float(max_depth_in), float(stepdown_in))
            path_elems.append(
                f"<path d=\"{outline_d}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.001\" "
                f"data-role=\"outline\" data-depth-in=\"{final_depth_q:.4f}\" />"
            )
        return _write_svg(path_elems, svg_w_m, svg_h_m, output_svg_path)

    base, ext = os.path.splitext(output_svg_path)
    layers: dict[float, List[str]] = {}
    if output_kind == "lines":
        for (orig_ln, elev_m), ln in zip(raw_lines, flipped_lines):
            depth_in = _depth_from_elevation_m(elev_m, float(zmin), float(zmax), float(max_depth_in))
            depth_q = _quantize_depth_in(depth_in, float(stepdown_in))
            d = _line_to_svg_path_open(list(ln.coords))
            elem = (
                f"<path d=\"{d}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.001\" "
                f"data-elev-m=\"{elev_m:.3f}\" data-depth-in=\"{depth_q:.4f}\" />"
            )
            layers.setdefault(depth_q, []).append(elem)
    else:
        if max_cut_in <= 0:
            layers.setdefault(0.0, [])
        else:
            band_step_in = float(thickness_in) / 8.0
            if band_step_in <= 0:
                band_step_in = float(max_cut_in)

            depth_marks: List[float] = [0.0]
            d = float(band_step_in)
            while d <= float(max_cut_in) + 1e-9:
                depth_marks.append(float(d))
                d += float(band_step_in)

            if depth_marks[-1] < float(max_cut_in) - 1e-9:
                depth_marks.append(float(max_cut_in))

            if len(depth_marks) >= 2:
                elev_thr = _elevation_from_depth_in(depth_marks[1], float(zmin), float(zmax), float(max_depth_in))
                lowlands = _pocket_region_below_threshold(z0, xs, ys, elev_thr)
                high_band = poly_target if lowlands is None else poly_target.difference(lowlands)
                if high_band is not None and (not high_band.is_empty):
                    minx, miny, _, _ = poly_target.bounds
                    hb_scaled = affinity.scale(high_band, xfact=scale, yfact=scale, origin=(minx, miny))
                    hb_scaled = affinity.translate(hb_scaled, xoff=xoff, yoff=yoff)
                    hb_flipped = affinity.scale(hb_scaled, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
                    hb_flipped = affinity.translate(hb_flipped, xoff=0.0, yoff=svg_h_m)
                    layers[0.0] = _geom_to_svg_paths_filled(hb_flipped)
                else:
                    layers.setdefault(0.0, [])
            else:
                layers.setdefault(0.0, [])

            prev_depth = 0.0
            for depth_q in depth_marks[1:]:
                elev_high = _elevation_from_depth_in(prev_depth, float(zmin), float(zmax), float(max_depth_in))
                elev_low = _elevation_from_depth_in(float(depth_q), float(zmin), float(zmax), float(max_depth_in))
                pocket_m = _pocket_region_in_band(z0, xs, ys, elev_low_m=elev_low, elev_high_m=elev_high)
                if pocket_m is None or pocket_m.is_empty:
                    layers.setdefault(float(depth_q), [])
                    prev_depth = float(depth_q)
                    continue
                pocket_m = pocket_m.intersection(poly_target)
                if pocket_m.is_empty:
                    layers.setdefault(float(depth_q), [])
                    prev_depth = float(depth_q)
                    continue

                minx, miny, _, _ = poly_target.bounds
                pocket_scaled = affinity.scale(pocket_m, xfact=scale, yfact=scale, origin=(minx, miny))
                pocket_scaled = affinity.translate(pocket_scaled, xoff=xoff, yoff=yoff)
                pocket_flipped = affinity.scale(pocket_scaled, xfact=1.0, yfact=-1.0, origin=(0.0, 0.0))
                pocket_flipped = affinity.translate(pocket_flipped, xoff=0.0, yoff=svg_h_m)

                layers[float(depth_q)] = _geom_to_svg_paths_filled(pocket_flipped)
                prev_depth = float(depth_q)

    out_paths: List[str] = []
    deepest_depth_q = max(layers.keys()) if layers else None
    for depth_q in sorted(layers.keys()):
        z_code = int(round(float(depth_q) * 10000.0))
        safe = f"z{z_code:04d}"
        elems = list(layers[depth_q])
        outline_d = _coords_to_svg_path(list(outline_flipped.exterior.coords))
        if include_outline_final_cut and deepest_depth_q is not None and depth_q == deepest_depth_q:
            elems.append(
                f"<path d=\"{outline_d}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.001\" "
                f"data-role=\"outline\" data-depth-in=\"{depth_q:.4f}\" />"
            )
        else:
            elems.append(
                f"<path d=\"{outline_d}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.001\" "
                f"data-role=\"frame\" data-depth-in=\"0.0000\" />"
            )
        out_paths.append(_write_svg(elems, svg_w_m, svg_h_m, f"{base}_{safe}{ext or '.svg'}"))

    if out_paths:
        return out_paths[0]

    outline_d = _coords_to_svg_path(list(outline_flipped.exterior.coords))
    return _write_svg(
        [
            f"<path d=\"{outline_d}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.001\" "
            f"data-role=\"frame\" data-depth-in=\"0.0000\" />"
        ],
        svg_w_m,
        svg_h_m,
        output_svg_path,
    )


def export_state_svg(
    state_query: str,
    target_width_in: float,
    target_height_in: float,
    output_svg_path: str,
    padding_in: float = 0.25,
    thickness_in: float = 0.75,
    smooth_iterations: int = 0,
) -> str:
    gdf = _load_states_gdf()
    row = _select_state(gdf, state_query)
    poly_m = _to_projected_meters(gdf, row)

    spec = MaterialSpec(
        target_width_in=target_width_in,
        target_height_in=target_height_in,
        padding_in=padding_in,
        thickness_in=thickness_in,
    )
    scaled, _, svg_w_m, svg_h_m, _, _ = _scale_to_material(poly_m, spec)

    exterior = list(scaled.exterior.coords)
    if smooth_iterations > 0:
        exterior = _chaikin_smooth_closed(exterior, iterations=smooth_iterations)
        scaled = Polygon(exterior)

    flipped = _flip_y_for_svg(scaled, svg_h_m)
    path_d = _coords_to_svg_path(list(flipped.exterior.coords))

    svg_w_in = svg_w_m / 0.0254
    svg_h_in = svg_h_m / 0.0254

    svg = (
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{svg_w_in:.4f}in\" height=\"{svg_h_in:.4f}in\" "
        f"viewBox=\"0 0 {svg_w_m:.6f} {svg_h_m:.6f}\">"
        f"<path d=\"{path_d}\" fill=\"none\" stroke=\"#000000\" stroke-width=\"0.001\" />"
        "</svg>"
    )

    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg)

    return output_svg_path


def _prompt_float(prompt: str, min_value: float) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            v = float(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if v < min_value:
            print(f"Enter a value >= {min_value}.")
            continue
        return v


def _prompt_str_default(prompt: str, default: str) -> str:
    raw = input(prompt).strip()
    return raw if raw != "" else default


def _sanitize_design_name(name: str) -> str:
    s = " ".join(name.strip().split())
    s = s.replace(os.sep, "-")
    return s


def _prompt_float_default(prompt: str, min_value: float, default: float) -> float:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            v = float(default)
        else:
            try:
                v = float(raw)
            except ValueError:
                print("Enter a number.")
                continue
        if v < min_value:
            print(f"Enter a value >= {min_value}.")
            continue
        return v


def _prompt_int(prompt: str, min_value: int) -> int:
    while True:
        raw = input(prompt).strip()
        try:
            v = int(raw)
        except ValueError:
            print("Enter an integer.")
            continue
        if v < min_value:
            print(f"Enter a value >= {min_value}.")
            continue
        return v


def _prompt_int_default(prompt: str, min_value: int, default: int) -> int:
    while True:
        raw = input(prompt).strip()
        if raw == "":
            v = int(default)
        else:
            try:
                v = int(raw)
            except ValueError:
                print("Enter an integer.")
                continue
        if v < min_value:
            print(f"Enter a value >= {min_value}.")
            continue
        return v


def main() -> None:
    state = _prompt_str_default(
        "Enter state name or abbreviation (e.g., CA or California) [Vermont]: ",
        "Vermont",
    )
    design_name = _sanitize_design_name(_prompt_str_default("Design name [vermont]: ", "vermont"))
    mode = (input("Export mode (outline/contours) [contours]: ").strip().lower() or "contours")
    width_in = _prompt_float_default("Enter wood block width in inches [11]: ", min_value=0.1, default=11.0)
    height_in = _prompt_float_default("Enter wood block height in inches [8]: ", min_value=0.1, default=8.0)
    thickness_in = _prompt_float_default("Enter wood block thickness in inches [0.75]: ", min_value=0.01, default=0.75)
    padding_in = _prompt_float_default("Enter padding in inches [0.25]: ", min_value=0.0, default=0.25)
    design_dir = os.path.join("outputs", design_name)
    out_path_default = os.path.join(design_dir, f"{design_name}.svg")
    out_path = _prompt_str_default(f"Output SVG path [{out_path_default}]: ", out_path_default)

    if mode == "outline":
        smooth_iters = _prompt_int("Enter smoothing iterations (0 for none, e.g., 2): ", min_value=0)
        export_state_svg(
            state_query=state,
            target_width_in=width_in,
            target_height_in=height_in,
            output_svg_path=out_path,
            padding_in=padding_in,
            thickness_in=thickness_in,
            smooth_iterations=smooth_iters,
        )
    else:
        dem_res = _prompt_int_default("Enter DEM resolution in meters (10/30/60) [30]: ", min_value=1, default=30)
        low_res = (input("Low-resolution 1km mode (max elevation per km)? (y/n) [n]: ").strip().lower() or "n") in {"y", "yes"}
        cache_tif_default = os.path.join(design_dir, "dem_cache.tif")
        cache_tif = _prompt_str_default(f"DEM cache GeoTIFF path [{cache_tif_default}]: ", cache_tif_default)
        interval_m = _prompt_float_default("Enter contour interval in meters [100]: ", min_value=0.1, default=100.0)
        sigma_px = _prompt_float_default("Enter smoothing sigma in pixels [1.5]: ", min_value=0.0, default=1.5)
        max_depth_in = _prompt_float_default(
            "Enter max carve depth in inches (<= thickness) [thickness]: ",
            min_value=0.0,
            default=float(thickness_in),
        )
        stepdown_in = _prompt_float_default("Enter stepdown per pass in inches [0.115]: ", min_value=0.0, default=0.115)
        per_layer = (input("Export one SVG per depth layer? (y/n) [y]: ").strip().lower() or "y") in {"y", "yes"}
        export_border = False
        border_svg_path = ""
        if per_layer:
            export_border = (input("Export border/outline as its own SVG file? (y/n) [y]: ").strip().lower() or "y") in {"y", "yes"}
            border_default = os.path.join(design_dir, f"{design_name}_border.svg")
            border_svg_path = _prompt_str_default(f"Border SVG path [{border_default}]: ", border_default) if export_border else ""
        output_kind = (input("Contour output kind (lines/pockets) [pockets]: ").strip().lower() or "pockets")
        export_state_contours_svg(
            state_query=state,
            target_width_in=width_in,
            target_height_in=height_in,
            output_svg_path=out_path,
            padding_in=padding_in,
            thickness_in=thickness_in,
            max_depth_in=max_depth_in,
            stepdown_in=stepdown_in,
            export_per_layer=per_layer,
            low_res_1km_max=low_res,
            low_res_cell_m=1000.0,
            dem_cache_tif=cache_tif,
            dem_resolution_m=int(dem_res),
            contour_interval_m=float(interval_m),
            smooth_sigma_px=float(sigma_px),
            output_kind=output_kind,
        )
        if export_border and border_svg_path:
            export_state_svg(
                state_query=state,
                target_width_in=width_in,
                target_height_in=height_in,
                output_svg_path=border_svg_path,
                padding_in=padding_in,
                thickness_in=thickness_in,
                smooth_iterations=0,
            )
    print(out_path)


if __name__ == "__main__":
    main()
