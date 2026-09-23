"""End-to-end job: region -> elevation -> layers -> SVG/STL/preview."""

from __future__ import annotations

import base64
import json
import re
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import MultiPolygon

from topology import boundary, mesh, svg
from topology.config import JobSpec
from topology.elevation import ElevationSource, TerrariumSource
from topology.layers import Layer, as_polys, build_layers
from topology.layout import BoardLayout, fit_layout
from topology.terrain import band_depths, prepare, region_mask, smooth_depth, thresholds

Progress = Callable[[str, float], None]

# Hypsometric tint, low -> high.
PALETTE = ["#1f4e5f", "#2a7f62", "#5fa35a", "#a9c25d", "#e0cf72", "#d9a05b", "#b86f4b", "#8f5a4a", "#c9c3bd", "#f5f3f0"]


def color_for(t: float) -> str:
    """t in [0,1] (0 = lowest) -> interpolated palette colour."""
    t = min(1.0, max(0.0, t)) * (len(PALETTE) - 1)
    i = min(int(t), len(PALETTE) - 2)
    f = t - i
    c0 = [int(PALETTE[i][j:j + 2], 16) for j in (1, 3, 5)]
    c1 = [int(PALETTE[i + 1][j:j + 2], 16) for j in (1, 3, 5)]
    return "#" + "".join(f"{round(a + (b - a) * f):02x}" for a, b in zip(c0, c1))


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "design"


@dataclass
class JobResult:
    spec: JobSpec
    name: str
    layout: BoardLayout
    layers: list[Layer]
    surround: MultiPolygon
    region_in: MultiPolygon
    zmin: float
    zmax: float
    stepped_height_in: np.ndarray
    smooth_height_in: np.ndarray
    surround_depth_in: Optional[float] = None
    timings: dict[str, float] = field(default_factory=dict)

    # ---------- summaries ----------
    def stats(self) -> dict:
        b = self.layout.board
        return {
            "name": self.name,
            "elevation_m": [self.zmin, self.zmax],
            "map_scale": round(self.layout.map_scale),
            "ground_px_m": self.layout.ground_px_m,
            "grid": list(self.layout.shape),
            "board_in": [b.width_in, b.height_in, b.thickness_in],
            "surround_depth_in": self.surround_depth_in,
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "layers": [
                {
                    "index": L.index,
                    "depth_in": round(L.depth_in, 4),
                    "elev_m": [round(L.elev_low_m, 1), round(L.elev_high_m, 1)],
                    "area_in2": round(L.area_in2, 3),
                    "color": self.layer_color(L.index),
                }
                for L in self.layers
            ],
        }

    def layer_color(self, k: int) -> str:
        n = len(self.layers)
        return color_for(1.0 - k / max(1, n - 1))

    # ---------- preview payload for the web UI ----------
    def preview(self, max_side: int = 400) -> dict:
        b = self.layout.board
        ny, nx = self.stepped_height_in.shape
        s = max(1, int(np.ceil(max(ny, nx) / max_side)))

        def enc(a: np.ndarray) -> str:
            q = np.clip(a / b.thickness_in, 0, 1) * 65535
            return base64.b64encode(q.astype("<u2").tobytes()).decode()

        st = self.stepped_height_in[::s, ::s]
        sm = self.smooth_height_in[::s, ::s]
        return {
            "stats": self.stats(),
            "heightmap": {"width": st.shape[1], "height": st.shape[0], "stepped": enc(st), "smooth": enc(sm)},
            "svg": {
                "layers": [svg.path_d(L.band) for L in self.layers],
                "surround": svg.path_d(self.surround) if not self.surround.is_empty else "",
                "outline": svg.path_d(self.region_in),
            },
        }

    # ---------- files ----------
    def layer_svg(self, L: Layer, cumulative: bool = False) -> str:
        b = self.layout.board
        geom = L.band
        if cumulative:
            # "Cut at least this deep": the region plus the (deeper) surround.
            geom = as_polys(L.region.union(self.surround))
        body = [svg.frame(b.width_in, b.height_in),
                svg.filled(geom, depth_in=f"{L.depth_in:.4f}", layer=L.index)]
        return svg.document(b.width_in, b.height_in, body,
                            f"{self.name} layer {L.index} depth {L.depth_in:.4f} in")

    def surround_svg(self) -> str:
        b = self.layout.board
        depth = self.surround_depth_in
        return svg.document(b.width_in, b.height_in,
                            [svg.frame(b.width_in, b.height_in), svg.filled(self.surround, depth_in=f"{depth:.4f}", role="surround")],
                            f"{self.name} surround depth {depth:.4f} in")

    def combined_svg(self) -> str:
        b = self.layout.board
        body = [f'<rect x="0" y="0" width="{b.width_in}" height="{b.height_in}" fill="#1b1d22"/>']
        for L in self.layers:
            body.append(f'<g id="layer-{L.index}" data-depth-in="{L.depth_in:.4f}">'
                        + svg.filled(L.band, self.layer_color(L.index)) + "</g>")
        body.append(svg.outline(self.region_in, "#ffffff"))
        return svg.document(b.width_in, b.height_in, body, self.name)

    def write(self, out_dir: str | Path, stl: bool = True, stl_max_pixels: int = 250_000) -> Path:
        out = Path(out_dir)
        (out / "cumulative").mkdir(parents=True, exist_ok=True)
        n = self.name
        files: list[Path] = []

        def put(fname: str, text: str) -> None:
            p = out / fname
            p.write_text(text, encoding="utf-8")
            files.append(p)

        put(f"{n}_combined.svg", self.combined_svg())
        b = self.layout.board
        put(f"{n}_outline.svg", svg.document(b.width_in, b.height_in,
                                             [svg.frame(b.width_in, b.height_in), svg.outline(self.region_in)],
                                             f"{n} outline"))
        # Layer 0 is the uncut surface, so only layers >= 1 produce pocket files.
        for L in self.layers[1:]:
            code = f"z{round(L.depth_in * 10000):04d}"
            put(f"{n}_layer{L.index:02d}_{code}.svg", self.layer_svg(L))
            put(f"cumulative/{n}_layer{L.index:02d}_{code}.svg", self.layer_svg(L, cumulative=True))
        if not self.surround.is_empty:
            put(f"{n}_surround.svg", self.surround_svg())

        if stl:
            for kind, h in (("stepped", self.stepped_height_in), ("smooth", self.smooth_height_in)):
                v, f = mesh.heightfield_mesh(mesh.downsample(h, stl_max_pixels), b.width_in, b.height_in)
                p = out / f"{n}_{kind}.stl"
                mesh.write_stl(p, v, f, f"{n} {kind}")
                files.append(p)

        manifest = {"spec": self.spec.to_dict(), "stats": self.stats(),
                    "files": [str(p.relative_to(out)) for p in files]}
        put("manifest.json", json.dumps(manifest, indent=2))

        zpath = out / f"{n}.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in files:
                z.write(p, p.relative_to(out))
        return zpath


def _resolve_region(spec: JobSpec) -> tuple[str, MultiPolygon]:
    if spec.state is not None:
        st = boundary.find_state(spec.state)
        name, geom = st.name, st.geometry
    else:
        name, geom = "custom region", boundary.geometry_from_geojson(spec.geojson)
    if spec.largest_only:
        geom = boundary.largest_part(geom)
    return spec.name or name, geom


def run_job(
    spec: JobSpec,
    source: Optional[ElevationSource] = None,
    progress: Optional[Progress] = None,
    region_wgs84: Optional[MultiPolygon] = None,
) -> JobResult:
    spec.validate()
    say = progress or (lambda msg, frac: None)
    timings: dict[str, float] = {}
    clock = time.perf_counter()

    def lap(key: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timings[key] = now - clock
        clock = now

    say("Resolving boundary", 0.02)
    if region_wgs84 is None:
        name, region_wgs84 = _resolve_region(spec)
    else:
        name = spec.name or "region"
    crs = boundary.local_crs(region_wgs84)
    region_m = boundary.project(region_wgs84, crs)
    layout = fit_layout(region_m, crs, spec.board)
    lap("boundary")

    say("Fetching elevation", 0.1)
    z_raw = (source or TerrariumSource()).sample(layout)
    lap("elevation")

    say("Analysing terrain", 0.45)
    ls = spec.layers
    mask = region_mask(region_m, layout)
    terrain = prepare(z_raw, mask, layout, ls)
    th = thresholds(terrain, ls)
    depths, surround_depth = band_depths(ls, ls.depth_in(spec.board))
    lap("terrain")

    say("Tracing layers", 0.6)
    region_in = layout.to_board(region_m)
    layers, surround = build_layers(
        terrain, th, depths, region_in, layout,
        surround_raised=surround_depth is not None,
        min_area_in2=ls.min_feature_in2, simplify_in=ls.simplify_in,
    )
    lap("vectorize")

    say("Building relief", 0.85)
    # Rasterise the final vectors so the 3D preview/STL match the SVGs exactly.
    px = Affine.scale(layout.px_w_in, layout.px_h_in)
    shapes = [(L.band, L.depth_in) for L in layers if not L.band.is_empty]
    if surround_depth is not None and not surround.is_empty:
        shapes.append((surround, surround_depth))
    depth = rasterize(shapes, out_shape=layout.shape, transform=px, fill=0.0, dtype="float64") if shapes \
        else np.zeros(layout.shape)
    t = spec.board.thickness_in
    stepped = t - depth
    sm = smooth_depth(terrain, th, depths)
    sm = np.where(mask, sm, surround_depth if surround_depth is not None else 0.0)
    smooth = t - sm
    lap("relief")
    say("Done", 1.0)

    return JobResult(
        spec=spec, name=slugify(name), layout=layout, layers=layers, surround=surround,
        region_in=region_in, zmin=terrain.zmin, zmax=terrain.zmax,
        stepped_height_in=stepped, smooth_height_in=smooth, surround_depth_in=surround_depth, timings=timings,
    )
