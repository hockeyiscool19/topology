"""Region boundaries: US states or arbitrary GeoJSON, and a local projection for each."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pyogrio
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform as shp_transform

from topology.cache import cache_dir, cached_download

# 1:500k cartographic boundaries: detailed enough for small states on a large board.
STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_state_500k.zip"

_lock = threading.Lock()


@dataclass(frozen=True)
class State:
    name: str
    usps: str
    geometry: MultiPolygon  # WGS84 lon/lat


@lru_cache(maxsize=1)
def load_states() -> dict[str, State]:
    """All states keyed by USPS code. Downloaded once, then read from a local GeoJSON cache."""
    with _lock:
        cached = cache_dir() / "states_500k_v2.geojson"
        if not cached.exists():
            z = cached_download(STATES_URL, cache_dir() / "cb_2023_us_state_500k.zip")
            meta, _, wkb, fields = pyogrio.raw.read(f"/vsizip/{z}", columns=["NAME", "STUSPS"])[:4]
            # Fields come back in file order, not the requested order.
            cols = list(meta["fields"])
            feats = []
            for i, g in enumerate(shapely.from_wkb(wkb)):
                props = {c: str(fields[j][i]) for j, c in enumerate(cols)}
                feats.append({"type": "Feature", "properties": props, "geometry": mapping(g)})
            tmp = cached.with_suffix(".part")
            tmp.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
            tmp.replace(cached)
        fc = json.loads(cached.read_text())
    out: dict[str, State] = {}
    for f in fc["features"]:
        p = f["properties"]
        out[p["STUSPS"].upper()] = State(p["NAME"], p["STUSPS"].upper(), as_multipolygon(shape(f["geometry"])))
    return out


def find_state(query: str) -> State:
    q = " ".join(query.strip().upper().split())
    states = load_states()
    if q in states:
        return states[q]
    for s in states.values():
        if s.name.upper() == q:
            return s
    matches = [s for s in states.values() if s.name.upper().startswith(q)] if q else []
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Unknown state: {query!r}")


def as_multipolygon(geom: Any) -> MultiPolygon:
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    if not polys:
        raise ValueError(f"Geometry has no polygon area: {geom.geom_type}")
    return as_multipolygon(shapely.union_all(polys))


def geometry_from_geojson(obj: dict[str, Any]) -> MultiPolygon:
    """Accept a Geometry, Feature or FeatureCollection (WGS84)."""
    t = obj.get("type")
    if t == "FeatureCollection":
        geoms = [shape(f["geometry"]) for f in obj.get("features", []) if f.get("geometry")]
        if not geoms:
            raise ValueError("FeatureCollection has no geometries")
        g = shapely.union_all(geoms)
    elif t == "Feature":
        g = shape(obj["geometry"])
    else:
        g = shape(obj)
    g = shapely.make_valid(g)
    mp = as_multipolygon(g)
    minx, miny, maxx, maxy = mp.bounds
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90:
        raise ValueError("GeoJSON must be WGS84 longitude/latitude")
    if mp.is_empty or mp.area <= 0:
        raise ValueError("Boundary has no area")
    return mp


def largest_part(mp: MultiPolygon) -> MultiPolygon:
    return MultiPolygon([max(mp.geoms, key=lambda p: p.area)])


def local_crs(geom_wgs84: MultiPolygon) -> CRS:
    """Albers equal-area conic fitted to the region: accurate shape for anything state-sized,
    and works anywhere on Earth (unlike a fixed national CRS)."""
    minx, miny, maxx, maxy = geom_wgs84.bounds
    # Regions crossing the antimeridian (e.g. Alaska's Aleutians) have huge lon spans; recentre.
    lon0 = _center_lon(geom_wgs84)
    lat0 = (miny + maxy) / 2
    span = maxy - miny
    lat1, lat2 = miny + span / 6, maxy - span / 6
    if abs(lat1 - lat2) < 1e-6:
        lat1, lat2 = lat0 - 0.5, lat0 + 0.5
    return CRS.from_proj4(
        f"+proj=aea +lat_0={lat0:.6f} +lon_0={lon0:.6f} +lat_1={lat1:.6f} +lat_2={lat2:.6f} "
        "+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )


def _center_lon(g: MultiPolygon) -> float:
    minx, _, maxx, _ = g.bounds
    if maxx - minx <= 180:
        return (minx + maxx) / 2
    # Shift negative longitudes by 360 and average, then wrap back.
    lons = [x if x >= 0 else x + 360 for p in g.geoms for x, _ in p.exterior.coords]
    c = (min(lons) + max(lons)) / 2
    return c - 360 if c > 180 else c


def project(geom_wgs84: MultiPolygon, crs: CRS) -> MultiPolygon:
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return as_multipolygon(shapely.make_valid(shp_transform(t.transform, geom_wgs84)))


def states_geojson(tolerance_deg: float = 0.01) -> dict[str, Any]:
    """Simplified states for the web map."""
    feats = []
    for s in load_states().values():
        g = s.geometry.simplify(tolerance_deg, preserve_topology=True)
        feats.append({"type": "Feature", "properties": {"name": s.name, "usps": s.usps}, "geometry": mapping(g)})
    return {"type": "FeatureCollection", "features": feats}
