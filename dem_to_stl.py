import os
from typing import Optional, Tuple

import numpy as np
import rioxarray as rxr
import trimesh


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


def _load_dem(dem_tif: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    da = rxr.open_rasterio(dem_tif, masked=True).squeeze(drop=True)

    if da.rio.crs is None:
        raise ValueError("DEM missing CRS")

    xs = np.asarray(da["x"].values, dtype=float)
    ys = np.asarray(da["y"].values, dtype=float)
    z = np.asarray(da.values, dtype=float)

    if z.ndim != 2:
        z = z.squeeze()
    if z.ndim != 2:
        raise ValueError("Unexpected DEM array shape")

    return xs, ys, z


def _downsample_grid(xs: np.ndarray, ys: np.ndarray, z: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = max(1, int(stride))
    xs2 = xs[::s]
    ys2 = ys[::s]
    z2 = z[::s, ::s]
    return xs2, ys2, z2


def _normalize_heights(z: np.ndarray) -> Tuple[np.ndarray, float, float]:
    z = np.asarray(z, dtype=float)
    z = np.where(np.isfinite(z), z, np.nan)
    zmin = float(np.nanmin(z))
    zmax = float(np.nanmax(z))
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError("DEM has invalid elevation range")

    zn = (z - zmin) / (zmax - zmin)
    return zn, zmin, zmax


def _quantize01(zn: np.ndarray, step: float) -> np.ndarray:
    if step <= 0:
        return zn
    return np.clip(np.round(zn / step) * step, 0.0, 1.0)


def _grid_to_mesh(xs_m: np.ndarray, ys_m: np.ndarray, z_in: np.ndarray, base_in: float, height_in: float) -> trimesh.Trimesh:
    # Units:
    # - x/y are meters from projected CRS, convert to mm
    # - z is in inches, convert to mm
    x_mm = xs_m * 1000.0
    y_mm = ys_m * 1000.0

    z_mm = (base_in + z_in * height_in) * 25.4

    ny, nx = z_mm.shape

    # Build vertices as row-major (y, x)
    xx, yy = np.meshgrid(x_mm, y_mm)
    verts = np.column_stack([xx.ravel(), yy.ravel(), z_mm.ravel()])

    # Two triangles per cell
    faces = []
    for j in range(ny - 1):
        row0 = j * nx
        row1 = (j + 1) * nx
        for i in range(nx - 1):
            a = row0 + i
            b = row0 + i + 1
            c = row1 + i
            d = row1 + i + 1
            faces.append([a, b, c])
            faces.append([b, d, c])

    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces, dtype=np.int64), process=False)

    # trimesh API differs across versions; use widely available cleanup ops.
    try:
        mesh.merge_vertices()
    except Exception:
        pass
    try:
        mesh.remove_degenerate_faces()
    except Exception:
        pass
    try:
        mesh.remove_unreferenced_vertices()
    except Exception:
        pass
    try:
        mesh.process(validate=True)
    except Exception:
        pass

    return mesh


def dem_to_stl(
    dem_tif: str,
    stl_out: str,
    mode: str,
    base_thickness_in: float,
    relief_height_in: float,
    stride: int,
    steps_in: Optional[float] = None,
) -> str:
    mode = (mode or "").strip().lower()
    if mode not in {"smooth", "stepped"}:
        raise ValueError("mode must be 'smooth' or 'stepped'")

    print("[dem_to_stl] ----")
    print(f"[dem_to_stl] Mode: {mode}")
    print(f"[dem_to_stl] Input DEM: {dem_tif}")
    print(f"[dem_to_stl] Output STL: {stl_out}")
    print(f"[dem_to_stl] Base thickness (in): {base_thickness_in}")
    print(f"[dem_to_stl] Relief height (in): {relief_height_in}")
    print(f"[dem_to_stl] Downsample stride: {stride}")

    print("[dem_to_stl] Step 1/6: Load DEM")
    xs, ys, z = _load_dem(dem_tif)
    print(f"[dem_to_stl]   Grid: {z.shape[1]} x {z.shape[0]} (nx x ny)")
    if xs.size >= 2 and ys.size >= 2:
        dx = abs(float(xs[1] - xs[0]))
        dy = abs(float(ys[1] - ys[0]))
        print(f"[dem_to_stl]   Resolution (m): dx={dx:.3f}, dy={dy:.3f}")

    print("[dem_to_stl] Step 2/6: Downsample grid")
    xs, ys, z = _downsample_grid(xs, ys, z, stride=stride)
    print(f"[dem_to_stl]   Grid after downsample: {z.shape[1]} x {z.shape[0]} (nx x ny)")

    print("[dem_to_stl] Step 3/6: Normalize elevations")
    zn, zmin, zmax = _normalize_heights(z)
    print(f"[dem_to_stl]   Elevation range (native units): min={zmin:.3f}, max={zmax:.3f}")

    approx_faces = int(max(0, (zn.shape[0] - 1) * (zn.shape[1] - 1) * 2))
    approx_verts = int(zn.shape[0] * zn.shape[1])
    print(f"[dem_to_stl]   Approx vertices: {approx_verts}")
    print(f"[dem_to_stl]   Approx faces (triangles): {approx_faces}")

    if mode == "stepped":
        if steps_in is None or steps_in <= 0:
            raise ValueError("steps_in must be > 0 for stepped mode")
        # Convert inch step height to [0..1] fraction of relief
        frac = float(steps_in) / float(relief_height_in) if relief_height_in > 0 else 0.0
        print("[dem_to_stl] Step 4/6: Quantize (stepped)")
        print(f"[dem_to_stl]   Step height (in): {steps_in}")
        print(f"[dem_to_stl]   Quantize fraction of relief: {frac:.6f}")
        zn = _quantize01(zn, step=frac)
    else:
        print("[dem_to_stl] Step 4/6: Quantize (stepped) - skipped (smooth mode)")

    print("[dem_to_stl] Step 5/6: Build mesh")
    mesh = _grid_to_mesh(xs, ys, zn, base_in=base_thickness_in, height_in=relief_height_in)
    try:
        print(f"[dem_to_stl]   Mesh vertices: {len(mesh.vertices)}")
        print(f"[dem_to_stl]   Mesh faces: {len(mesh.faces)}")
    except Exception:
        pass

    print("[dem_to_stl] Step 6/6: Export STL")
    os.makedirs(os.path.dirname(stl_out) or ".", exist_ok=True)
    try:
        mesh.export(stl_out, file_type="stl", binary=True)
    except TypeError:
        mesh.export(stl_out)
    try:
        size_mb = os.path.getsize(stl_out) / (1024.0 * 1024.0)
        print(f"[dem_to_stl]   Wrote: {stl_out} ({size_mb:.2f} MB)")
    except Exception:
        print(f"[dem_to_stl]   Wrote: {stl_out}")
    return stl_out


def main() -> None:
    dem_in = _prompt_str_default(
        "Input DEM GeoTIFF path [outputs/vermont/dem_cache.tif]: ",
        "outputs/vermont/dem_cache.tif",
    )

    base_in = _prompt_float_default("Base thickness in inches [0.2]: ", min_value=0.0, default=0.2)
    relief_in = _prompt_float_default("Relief height in inches [0.5]: ", min_value=0.0, default=0.5)
    stride = _prompt_int_default("Downsample stride (1=no downsample) [4]: ", min_value=1, default=4)

    base, _ = os.path.splitext(dem_in)
    smooth_out_default = base + "_smooth.stl"
    stepped_out_default = base + "_stepped.stl"

    smooth_out = _prompt_str_default(f"Smooth STL output [{smooth_out_default}]: ", smooth_out_default)
    stepped_out = _prompt_str_default(f"Stepped STL output [{stepped_out_default}]: ", stepped_out_default)

    step_in = _prompt_float_default("Stepped layer height in inches [0.0625]: ", min_value=0.0001, default=0.0625)

    dem_to_stl(
        dem_tif=dem_in,
        stl_out=smooth_out,
        mode="smooth",
        base_thickness_in=base_in,
        relief_height_in=relief_in,
        stride=stride,
    )
    dem_to_stl(
        dem_tif=dem_in,
        stl_out=stepped_out,
        mode="stepped",
        base_thickness_in=base_in,
        relief_height_in=relief_in,
        stride=stride,
        steps_in=step_in,
    )

    print(smooth_out)
    print(stepped_out)


if __name__ == "__main__":
    main()
