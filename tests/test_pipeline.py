import json
import zipfile

import numpy as np
import pytest
import shapely
import trimesh
from shapely.geometry import Point, box

from topology import BoardSpec, JobSpec, LayerSpec, run_job
from topology.config import BoardSpec as BS
from topology.layers import region_below
from topology.layout import fit_layout
from topology.mesh import heightfield_mesh
from topology.terrain import band_depths

# A ~50 km square near Vermont.
SQUARE = {"type": "Polygon", "coordinates": [[[-73.0, 44.0], [-72.4, 44.0], [-72.4, 44.45], [-73.0, 44.45], [-73.0, 44.0]]]}


class Synthetic:
    """Crater (ring ridge around a deep bowl) plus a separate peak: exercises holes and islands."""

    def sample(self, layout):
        ny, nx = layout.shape
        cols, rows = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5)
        t = layout.transform
        x, y = t.c + t.a * cols, t.f + t.e * rows
        cx, cy = layout.center_m
        r = np.hypot(x - cx + 8000, y - cy)
        crater = 900 * np.exp(-((r - 9000) / 3500) ** 2) - 300 * np.exp(-(r / 3000) ** 2)
        peak = 1200 * np.exp(-(np.hypot(x - cx - 16000, y - cy - 9000) / 4000) ** 2)
        z = 200 + crater + peak
        z[0, 0] = np.nan  # nodata must not break anything
        return z.astype(np.float32)


def job(**layers):
    return JobSpec(geojson=SQUARE, name="test", board=BoardSpec(width_in=8, height_in=6, resolution_in=0.02),
                   layers=LayerSpec(**layers))


@pytest.fixture(scope="module")
def result():
    return run_job(job(count=6), source=Synthetic())


def test_bands_disjoint_and_tile_region(result):
    bands = [L.band for L in result.layers]
    for i in range(len(bands)):
        for j in range(i + 1, len(bands)):
            assert shapely.intersection(bands[i], bands[j]).area < 1e-6, (i, j)
    total = sum(b.area for b in bands)
    assert total == pytest.approx(result.region_in.area, rel=1e-6)
    assert shapely.union_all(bands).symmetric_difference(result.region_in).area < 1e-6


def test_regions_nested(result):
    regs = [L.region for L in result.layers]
    for a, b in zip(regs, regs[1:]):
        assert b.difference(a).area < 1e-9


def test_depths_monotonic_and_bounded(result):
    d = [L.depth_in for L in result.layers]
    assert d[0] == 0 and all(b > a for a, b in zip(d, d[1:]))
    b = result.layout.board
    assert result.surround_depth_in == pytest.approx(b.max_cut_in)
    assert result.stepped_height_in.min() >= b.floor_in - 1e-9
    assert result.stepped_height_in.max() <= b.thickness_in + 1e-9
    assert np.isfinite(result.smooth_height_in).all()


def test_crater_bowl_is_a_hole_not_filled(result):
    # The bowl in the crater's centre is low, the ring around it high: the highest band that
    # contains the ring must have a hole (or the bowl must be in a deeper band).
    lay = result.layout
    cx, cy = lay.center_m
    bowl = lay.to_board(Point(cx - 8000, cy))
    ring = lay.to_board(Point(cx - 8000 + 9000, cy))
    band_of = lambda p: next(L.index for L in result.layers if L.band.buffer(1e-9).contains(p))
    assert band_of(bowl) > band_of(ring)


def test_design_is_centered_with_equal_padding(result):
    b = result.layout.board
    minx, miny, maxx, maxy = result.region_in.bounds
    assert minx == pytest.approx(b.width_in - maxx, abs=1e-6)
    assert miny == pytest.approx(b.height_in - maxy, abs=1e-6)
    assert min(minx, miny) == pytest.approx(b.padding_in, abs=1e-6)


def test_frame_mode_has_no_surround():
    r = run_job(job(count=4, surround="frame"), source=Synthetic())
    assert r.surround.is_empty and r.surround_depth_in is None
    assert r.layers[-1].depth_in == pytest.approx(r.layout.board.max_cut_in)


@pytest.mark.parametrize("curve,ex", [("equal-area", 1.0), ("linear", 2.0), ("linear", 0.5)])
def test_curves(curve, ex):
    r = run_job(job(count=5, curve=curve, exaggeration=ex), source=Synthetic())
    elev = [L.elev_high_m for L in r.layers]
    assert all(a >= b for a, b in zip(elev, elev[1:]))
    if curve == "equal-area":
        areas = np.array([L.area_in2 for L in r.layers])
        assert areas.std() / areas.mean() < 0.35


def test_write_outputs(tmp_path, result):
    z = result.write(tmp_path)
    names = zipfile.ZipFile(z).namelist()
    assert "manifest.json" in names
    assert sum(n.startswith("test_layer") for n in names) == len(result.layers) - 1
    assert "test_surround.svg" in names
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert len(m["stats"]["layers"]) == 6
    for kind in ("stepped", "smooth"):
        mesh = trimesh.load(tmp_path / f"test_{kind}.stl")
        assert mesh.is_watertight and mesh.volume > 0
        ext = mesh.bounds[1] - mesh.bounds[0]
        assert ext == pytest.approx([8 * 25.4, 6 * 25.4, 0.75 * 25.4], rel=0.02)
    code = round(result.layers[1].depth_in * 10000)
    svg = (tmp_path / f"test_layer01_z{code:04d}.svg").read_text()
    assert 'width="8.0000in"' in svg and 'viewBox="0 0 8.0000 6.0000"' in svg


def test_mesh_watertight_small():
    h = np.random.default_rng(0).uniform(0.3, 0.75, (7, 9))
    v, f = heightfield_mesh(h, 3, 2)
    m = trimesh.Trimesh(v, f, process=True)
    assert m.is_watertight and m.is_winding_consistent and m.volume > 0


def test_region_below_handles_nested_islands():
    lay = fit_layout(shapely.MultiPolygon([box(0, 0, 100, 100)]), None, BS(width_in=4, height_in=4, padding_in=0, resolution_in=0.05))
    z = np.ones(lay.shape)
    z[10:70, 10:70] = 0  # low square
    z[25:55, 25:55] = 1  # high island inside it
    z[35:45, 35:45] = 0  # low lake inside the island
    r = region_below(z, 0.5, lay, 0)
    p = 0.05
    area = (60 * 60 - 30 * 30 + 10 * 10) * p * p
    assert r.area == pytest.approx(area, rel=0.03)


def test_band_depths():
    d, s = band_depths(LayerSpec(count=4, surround="raised"), 0.5)
    assert s == 0.5 and np.allclose(d, [0, 0.125, 0.25, 0.375])
    d, s = band_depths(LayerSpec(count=5, surround="frame"), 0.5)
    assert s is None and np.allclose(d, [0, 0.125, 0.25, 0.375, 0.5])


def test_validation_errors():
    with pytest.raises(ValueError):
        JobSpec(state="VT", geojson=SQUARE).validate()
    with pytest.raises(ValueError):
        job(count=1).validate()
    with pytest.raises(ValueError):
        JobSpec(state="VT", layers=LayerSpec(max_depth_in=0.7)).validate()  # > thickness - floor


def test_spec_roundtrip():
    s = job(count=7, curve="equal-area")
    assert JobSpec.from_dict(json.loads(json.dumps(s.to_dict()))) == s


def test_even_odd_deep_nesting():
    from topology.layers import even_odd
    rings = [box(0, 0, 10, 10), box(1, 1, 9, 9), box(2, 2, 8, 8), box(3, 3, 7, 7), box(20, 0, 22, 2)]
    g = even_odd(rings)
    assert g.area == pytest.approx(100 - 64 + 36 - 16 + 4)
    assert g.contains(Point(0.5, 0.5)) and not g.contains(Point(1.5, 1.5))
    assert g.contains(Point(2.5, 2.5)) and not g.contains(Point(5, 5))
