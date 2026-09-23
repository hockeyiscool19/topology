"""SVG output in real board units (inches), ready for Easel / Carbide Create / VCarve."""

from __future__ import annotations

from html import escape
from typing import Iterable, Sequence

from shapely.geometry import MultiPolygon, Polygon


def _ring_d(coords: Sequence[tuple[float, float]]) -> str:
    pts = list(coords)
    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) < 3:
        return ""
    head = f"M{pts[0][0]:.4f} {pts[0][1]:.4f}"
    return head + "".join(f"L{x:.4f} {y:.4f}" for x, y in pts[1:]) + "Z"


def path_d(geom: MultiPolygon | Polygon) -> str:
    """SVG path data for polygons with holes (use fill-rule=evenodd)."""
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    parts: list[str] = []
    for p in polys:
        parts.append(_ring_d(p.exterior.coords))
        parts.extend(_ring_d(r.coords) for r in p.interiors)
    return "".join(parts)


def document(width_in: float, height_in: float, body: Iterable[str], title: str = "") -> str:
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_in:.4f}in" height="{height_in:.4f}in" '
        f'viewBox="0 0 {width_in:.4f} {height_in:.4f}">'
    )
    t = f"<title>{escape(title)}</title>" if title else ""
    return head + t + "".join(body) + "</svg>\n"


def frame(width_in: float, height_in: float) -> str:
    """Board-sized registration rectangle so every layer file imports at the same position."""
    return (
        f'<rect data-role="frame" x="0" y="0" width="{width_in:.4f}" height="{height_in:.4f}" '
        'fill="none" stroke="#999999" stroke-width="0.005"/>'
    )


def filled(geom: MultiPolygon, fill: str = "#000000", **data: object) -> str:
    if geom.is_empty:
        return ""
    attrs = "".join(f' data-{k.replace("_", "-")}="{escape(str(v))}"' for k, v in data.items())
    return f'<path d="{path_d(geom)}" fill="{fill}" fill-rule="evenodd" stroke="none"{attrs}/>'


def outline(geom: MultiPolygon, stroke: str = "#000000") -> str:
    if geom.is_empty:
        return ""
    return f'<path data-role="outline" d="{path_d(geom)}" fill="none" stroke="{stroke}" stroke-width="0.01"/>'
