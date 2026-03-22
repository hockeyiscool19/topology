import argparse
import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union
from svgpathtools import svg2paths2


_Z_RE = re.compile(r"(?:^|_)z(\d{4})(?:\.|$)")


_USPS_TO_DIR = {
    "AL": "alabama",
    "AK": "alaska",
    "AZ": "arizona",
    "AR": "arkansas",
    "CA": "california",
    "CO": "colorado",
    "CT": "connecticut",
    "DE": "delaware",
    "FL": "florida",
    "GA": "georgia",
    "HI": "hawaii",
    "ID": "idaho",
    "IL": "illinois",
    "IN": "indiana",
    "IA": "iowa",
    "KS": "kansas",
    "KY": "kentucky",
    "LA": "louisiana",
    "ME": "maine",
    "MD": "maryland",
    "MA": "massachusetts",
    "MI": "michigan",
    "MN": "minnesota",
    "MS": "mississippi",
    "MO": "missouri",
    "MT": "montana",
    "NE": "nebraska",
    "NV": "nevada",
    "NH": "new_hampshire",
    "NJ": "new_jersey",
    "NM": "new_mexico",
    "NY": "new_york",
    "NC": "north_carolina",
    "ND": "north_dakota",
    "OH": "ohio",
    "OK": "oklahoma",
    "OR": "oregon",
    "PA": "pennsylvania",
    "RI": "rhode_island",
    "SC": "south_carolina",
    "SD": "south_dakota",
    "TN": "tennessee",
    "TX": "texas",
    "UT": "utah",
    "VT": "vermont",
    "VA": "virginia",
    "WA": "washington",
    "WV": "west_virginia",
    "WI": "wisconsin",
    "WY": "wyoming",
    "DC": "district_of_columbia",
}


def _resolve_outputs_dir(token: str) -> Optional[str]:
    t0 = token.strip()
    base = os.path.join("outputs", t0)
    if os.path.isdir(base):
        return base

    if len(t0) == 2:
        mapped = _USPS_TO_DIR.get(t0.upper())
        if mapped is not None:
            base2 = os.path.join("outputs", mapped)
            if not os.path.exists(base2):
                os.makedirs(base2, exist_ok=True)
            if os.path.isdir(base2):
                return base2

    if not os.path.isdir("outputs"):
        return None

    t = t0.lower()
    if not t:
        return None

    candidates = []
    for p in glob.glob(os.path.join("outputs", "*")):
        if not os.path.isdir(p):
            continue
        name = os.path.basename(p).strip().lower()
        if name.startswith(t):
            candidates.append(p)

    if len(candidates) == 1:
        return candidates[0]

    return None


def _depth_in_from_filename(path: str) -> Optional[float]:
    m = _Z_RE.search(os.path.basename(path))
    if not m:
        return None
    return int(m.group(1)) / 10000.0


def _path_to_polygon(path, samples: int) -> Optional[Polygon]:
    if not path.isclosed():
        return None

    pts = []
    for t in np.linspace(0.0, 1.0, int(samples), endpoint=False):
        z = path.point(float(t))
        pts.append((float(z.real), float(z.imag)))

    if len(pts) < 3:
        return None

    p = Polygon(pts)
    if p.is_empty or p.area <= 0:
        return None
    if not p.is_valid:
        p = p.buffer(0)
    return p if (p is not None and (not p.is_empty) and p.area > 0) else None


def load_layer_geometry(svg_path: str, samples: int = 256):
    paths, attributes, _svg_attributes = svg2paths2(svg_path)

    polys: List[Polygon] = []
    for path, attr in zip(paths, attributes):
        fill = (attr.get("fill") or "").strip().lower()
        if fill in {"", "none"}:
            continue

        for sp in path.continuous_subpaths():
            poly = _path_to_polygon(sp, samples=samples)
            if poly is not None:
                polys.append(poly)

    if not polys:
        return None

    merged = unary_union(polys)
    if merged is None or merged.is_empty:
        return None

    if not merged.is_valid:
        merged = merged.buffer(0)

    return merged if (merged is not None and (not merged.is_empty)) else None


def _expand_inputs(inputs: List[str]) -> List[str]:
    expanded: List[str] = []
    for p in inputs:
        matches = glob.glob(p)
        if matches:
            expanded.extend(matches)
            continue

        if os.path.isdir(p):
            expanded.extend(glob.glob(os.path.join(p, "*_z*.svg")))
            continue

        if not os.path.exists(p) and ("/" not in p) and (not p.lower().endswith(".svg")):
            resolved = _resolve_outputs_dir(p)
            if resolved is not None:
                expanded.extend(glob.glob(os.path.join(resolved, "*_z*.svg")))
            continue

        expanded.append(p)

    return sorted({p for p in expanded if p.lower().endswith(".svg")})


def validate_no_overlap(svg_paths: List[str], samples: int = 256, area_tol: float = 1e-6) -> Tuple[bool, str]:
    layers: Dict[float, List[str]] = {}
    for sp in svg_paths:
        d = _depth_in_from_filename(sp)
        if d is None:
            continue
        layers.setdefault(float(d), []).append(sp)

    if not layers:
        return False, "No layer SVGs found (expected filenames containing _z####.svg)."

    depth_sorted = sorted(layers.keys())

    report_lines: List[str] = []
    ok = True

    cumulative = None
    for depth in depth_sorted:
        geoms = []
        for sp in layers[depth]:
            g = load_layer_geometry(sp, samples=samples)
            if g is not None and (not g.is_empty):
                geoms.append(g)

        layer_geom = unary_union(geoms) if geoms else None
        if layer_geom is not None and (not layer_geom.is_empty) and (not layer_geom.is_valid):
            layer_geom = layer_geom.buffer(0)

        overlap_area = 0.0
        if cumulative is not None and layer_geom is not None and (not layer_geom.is_empty):
            try:
                inter = layer_geom.intersection(cumulative)
                overlap_area = float(inter.area) if (inter is not None and (not inter.is_empty)) else 0.0
            except Exception:
                overlap_area = float("nan")

        z_code = int(round(float(depth) * 10000.0))
        report_lines.append(f"z{z_code:04d} depth_in={depth:.4f} overlap_area={overlap_area:.6f}")

        if np.isfinite(overlap_area) and overlap_area > float(area_tol):
            ok = False

        if layer_geom is not None and (not layer_geom.is_empty):
            cumulative = layer_geom if cumulative is None else cumulative.union(layer_geom)

    if ok:
        return True, "No overlaps detected.\n" + "\n".join(report_lines)
    return False, "Overlaps detected!\n" + "\n".join(report_lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="+",
        help=(
            "Layer SVGs, a directory, or a state folder name. "
            "Examples: 'vermont' OR 'outputs/vermont' OR 'outputs/vermont/*_z*.svg'"
        ),
    )
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--area-tol", type=float, default=1e-6)
    args = ap.parse_args()

    svg_paths = _expand_inputs(list(args.paths))
    ok, msg = validate_no_overlap(svg_paths, samples=int(args.samples), area_tol=float(args.area_tol))
    print(msg)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
