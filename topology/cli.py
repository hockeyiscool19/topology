"""Command line: `topology generate ...` and `topology serve`."""

from __future__ import annotations

import argparse
import json
import os
import sys

from topology.config import BoardSpec, JobSpec, LayerSpec


def _generate(a: argparse.Namespace) -> int:
    geojson = None
    if a.geojson:
        with open(a.geojson, encoding="utf-8") as f:
            geojson = json.load(f)
    spec = JobSpec(
        state=a.state, geojson=geojson, name=a.name, largest_only=a.largest_only,
        board=BoardSpec(width_in=a.width, height_in=a.height, thickness_in=a.thickness,
                        padding_in=a.padding, floor_in=a.floor, resolution_in=a.resolution),
        layers=LayerSpec(count=a.layers, max_depth_in=a.max_depth, smoothing_in=a.smoothing,
                         curve=a.curve, exaggeration=a.exaggeration, min_feature_in2=a.min_feature,
                         surround=a.surround),
    )
    from topology.pipeline import run_job

    def progress(msg: str, frac: float) -> None:
        print(f"[{frac * 100:5.1f}%] {msg}", file=sys.stderr)

    try:
        result = run_job(spec, progress=progress)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = a.out or f"outputs/{result.name}"
    z = result.write(out, stl=not a.no_stl)
    s = result.stats()
    print(f"{s['name']}: elevation {s['elevation_m'][0]:.0f}–{s['elevation_m'][1]:.0f} m, "
          f"scale 1:{s['map_scale']:,}", file=sys.stderr)
    for L in s["layers"]:
        print(f"  layer {L['index']:2d}  depth {L['depth_in']:.4f} in  "
              f"elev {L['elev_m'][0]:7.1f}–{L['elev_m'][1]:7.1f} m  area {L['area_in2']:7.2f} in²", file=sys.stderr)
    print(z)
    return 0


def _serve(a: argparse.Namespace) -> int:
    import uvicorn

    from topology.server import create_app

    uvicorn.run(create_app(), host=a.host, port=a.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="topology", description="Terrain -> layered CNC carving files")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate SVG layers + STL for a region")
    src = g.add_mutually_exclusive_group(required=True)
    src.add_argument("--state", help="US state name or code, e.g. VT or 'North Carolina'")
    src.add_argument("--geojson", help="path to a GeoJSON boundary (WGS84)")
    g.add_argument("--name")
    g.add_argument("--out", help="output directory [outputs/<name>]")
    b, l = BoardSpec(), LayerSpec()
    g.add_argument("--width", type=float, default=b.width_in, help="board width, in")
    g.add_argument("--height", type=float, default=b.height_in, help="board height, in")
    g.add_argument("--thickness", type=float, default=b.thickness_in, help="board thickness, in")
    g.add_argument("--padding", type=float, default=b.padding_in, help="margin around the design, in")
    g.add_argument("--floor", type=float, default=b.floor_in, help="uncut material at the bottom, in")
    g.add_argument("--resolution", type=float, default=b.resolution_in, help="raster resolution, in/px")
    g.add_argument("--layers", type=int, default=l.count, help="number of heights incl. the top surface")
    g.add_argument("--max-depth", type=float, default=None, help="deepest cut, in [thickness - floor]")
    g.add_argument("--smoothing", type=float, default=l.smoothing_in, help="terrain blur radius, in")
    g.add_argument("--curve", choices=["linear", "equal-area"], default=l.curve)
    g.add_argument("--exaggeration", type=float, default=l.exaggeration, help=">1 emphasises peaks")
    g.add_argument("--min-feature", type=float, default=l.min_feature_in2, help="drop specks smaller than this, in²")
    g.add_argument("--surround", choices=["raised", "frame"], default=l.surround)
    g.add_argument("--largest-only", action="store_true", help="drop islands / secondary parts")
    g.add_argument("--no-stl", action="store_true")
    g.set_defaults(fn=_generate)

    s = sub.add_parser("serve", help="run the web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    s.set_defaults(fn=_serve)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
