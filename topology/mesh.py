"""Watertight relief meshes from a height field, fully vectorised."""

from __future__ import annotations

import numpy as np

MM_PER_IN = 25.4


def heightfield_mesh(height_in: np.ndarray, width_in: float, depth_in: float) -> tuple[np.ndarray, np.ndarray]:
    """Closed solid: height field on top, vertical walls, flat base at z=0. Units: mm.

    height_in[row, col] with row 0 at the board's top edge (SVG convention). The mesh uses
    y up, so row 0 maps to the largest y.
    """
    h = np.asarray(height_in, dtype=np.float64)
    ny, nx = h.shape
    if ny < 2 or nx < 2:
        raise ValueError("Height field must be at least 2x2")
    if not np.all(np.isfinite(h)):
        raise ValueError("Height field contains non-finite values")

    xs = np.linspace(0.0, width_in, nx) * MM_PER_IN
    ys = np.linspace(depth_in, 0.0, ny) * MM_PER_IN
    xx, yy = np.meshgrid(xs, ys)
    top = np.column_stack([xx.ravel(), yy.ravel(), h.ravel() * MM_PER_IN])

    idx = np.arange(ny * nx).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    # Rows go downward in y, so (a, c, b) is counter-clockwise seen from +z.
    top_faces = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])

    # Perimeter, counter-clockwise seen from above: bottom edge L->R, right edge up,
    # top edge R->L, left edge down.
    ring = np.concatenate([
        idx[-1, :],
        idx[-2::-1, -1],
        idx[0, -2::-1],
        idx[1:-1, 0],
    ])
    n_top = top.shape[0]
    m = ring.size
    bottom = top[ring].copy()
    bottom[:, 2] = 0.0
    bot = n_top + np.arange(m)
    center = n_top + m
    verts = np.vstack([top, bottom, [[width_in * MM_PER_IN / 2, depth_in * MM_PER_IN / 2, 0.0]]])

    t0, t1 = ring, np.roll(ring, -1)
    b0, b1 = bot, np.roll(bot, -1)
    walls = np.concatenate([np.column_stack([t0, b0, b1]), np.column_stack([t0, b1, t1])])
    base = np.column_stack([np.full(m, center), b1, b0])

    faces = np.concatenate([top_faces, walls, base]).astype(np.int64)
    return verts, faces


def write_stl(path, verts: np.ndarray, faces: np.ndarray, name: str = "topology") -> None:
    tri = verts[faces].astype(np.float32)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = tri
    header = name.encode("ascii", "replace")[:80].ljust(80, b" ")
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.uint32(len(faces)).tobytes())
        f.write(rec.tobytes())


def downsample(a: np.ndarray, max_pixels: int) -> np.ndarray:
    """Nearest-neighbour stride (keeps stepped edges crisp)."""
    ny, nx = a.shape
    s = max(1, int(np.ceil(np.sqrt(ny * nx / max_pixels))))
    return a[::s, ::s]
