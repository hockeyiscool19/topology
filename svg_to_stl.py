import os
from typing import List

import numpy as np
import trimesh
from svgpathtools import svg2paths2


def _prompt_str_default(prompt: str, default: str) -> str:
    raw = input(prompt).strip()
    return raw if raw != "" else default


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


def _sample_path(poly_path, samples_per_segment: int) -> np.ndarray:
    n = max(2, int(samples_per_segment))
    pts: List[complex] = []
    for seg in poly_path:
        ts = np.linspace(0.0, 1.0, n, endpoint=False)
        pts.extend([seg.point(float(t)) for t in ts])
    pts.append(poly_path[-1].point(1.0))
    arr = np.array([[p.real, p.imag] for p in pts], dtype=float)
    return arr


def _ensure_closed(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        raise ValueError("Not enough points")
    if not np.allclose(points[0], points[-1]):
        points = np.vstack([points, points[0]])
    return points


def _polygon_area(points: np.ndarray) -> float:
    pts = points
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def svg_outline_to_stl(
    svg_path: str,
    stl_path: str,
    thickness_in: float,
    samples_per_segment: int = 30,
) -> str:
    paths, attributes, svg_attr = svg2paths2(svg_path)
    if not paths:
        raise ValueError("No paths found in SVG")

    best = None
    best_area = 0.0
    for p in paths:
        pts = _ensure_closed(_sample_path(p, samples_per_segment=samples_per_segment))
        area = abs(_polygon_area(pts))
        if area > best_area:
            best_area = area
            best = pts

    if best is None:
        raise ValueError("Could not extract a closed outline")

    pts = best[:-1]
    if pts.shape[0] < 3:
        raise ValueError("Outline too small")

    poly = trimesh.path.polygons.polygon.Polygon(pts)
    if poly.is_empty:
        raise ValueError("Invalid polygon derived from SVG")

    height_mm = float(thickness_in) * 25.4
    mesh = trimesh.creation.extrude_polygon(poly, height=height_mm)

    os.makedirs(os.path.dirname(stl_path) or ".", exist_ok=True)
    mesh.export(stl_path)
    return stl_path


def main() -> None:
    svg_in = _prompt_str_default("Input SVG path [outputs/vermont/vermont.svg]: ", "outputs/vermont/vermont.svg")
    out_default = os.path.splitext(svg_in)[0] + ".stl"
    stl_out = _prompt_str_default(f"Output STL path [{out_default}]: ", out_default)
    thickness_in = _prompt_float_default("Extrusion thickness in inches [0.75]: ", min_value=0.01, default=0.75)
    samples = int(_prompt_float_default("Samples per segment [30]: ", min_value=2, default=30))

    svg_outline_to_stl(svg_in, stl_out, thickness_in=thickness_in, samples_per_segment=samples)
    print(stl_out)


if __name__ == "__main__":
    main()
