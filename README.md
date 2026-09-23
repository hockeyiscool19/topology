# Topology

Turn any landscape into CNC-ready carving files. Pick a US state (or draw any area on the map), set your board size and number of layers, and Topology slices real elevation data into clean, **non-overlapping pocket layers** (SVG) plus a **watertight STL**, with a live 3D preview of the carved board.

![C06F2323-1F90-4C62-903F-07D728E7B05C_1_105_c](https://github.com/user-attachments/assets/7f0a4e68-f8b5-456d-b6b0-f8935fdc2826)

## Quick start

```bash
# backend (Python ≥ 3.10)
uv venv && uv pip install -e ".[dev]"

# frontend
cd web && npm install && npm run build && cd ..

# run the app at http://127.0.0.1:8000
.venv/bin/topology serve
```

For frontend development, run `topology serve` and `cd web && npm run dev` (Vite on :5173 proxies `/api` to :8000; set `TOPOLOGY_API` to point elsewhere).

### Command line

```bash
topology generate --state VT --width 11 --height 8 --layers 8
topology generate --geojson my_area.geojson --name adirondacks --curve equal-area --surround frame
topology generate --help   # all options
```

Outputs land in `outputs/<name>/`:

| File | What it is |
| --- | --- |
| `<name>_layerNN_zDDDD.svg` | One pocket per depth (`zDDDD` = depth × 10 000 in). **Disjoint**: every point on the board belongs to exactly one layer. |
| `cumulative/…` | Same layers as "cut at least this deep" regions (nested). Use these if your CAM prefers stacked pockets. |
| `<name>_surround.svg` | Area outside the region (raised mode), cut to full depth. |
| `<name>_outline.svg` | Region outline, for a profile cut. |
| `<name>_combined.svg` | Colour preview of all layers. |
| `<name>_stepped.stl` / `_smooth.stl` | Watertight relief meshes in mm (3D print, CAM 3D carving, or visual check). |
| `manifest.json`, `<name>.zip` | Settings, per-layer stats, everything zipped. |

Every SVG is exactly board-sized, in inches, with a registration frame, so layers import into Easel / Carbide Create / VCarve at the same position.

## How it works

1. **Boundary.** A US state (Census 1:500k cartographic boundaries) or any GeoJSON polygon, projected with an Albers equal-area projection fitted to the region. That works anywhere on Earth, including across the antimeridian (Alaska).
2. **Fit.** The region is scaled to the board minus padding and centred. A single affine transform maps board pixels to map metres.
3. **Elevation.** AWS Terrain Tiles are fetched at the zoom that matches the board's resolution, cached on disk, and warped straight onto the board grid in one resample. Work scales with board size, not region size, so California costs about the same as Vermont.
4. **Terrain.** Gaussian smoothing uses a radius in board inches and normalised convolution (the boundary doesn't drag elevations down). Thresholds are either equal-height or equal-area, with an optional peak/lowland emphasis curve.
5. **Layers.** Marching squares traces each threshold with sub-pixel smooth edges. Rings are combined with the even-odd rule, so valleys inside ridges stay holes and peaks inside valleys stay islands. Regions are then clipped so they're strictly nested, and bands are their differences, so they're disjoint by construction. Specks smaller than the min feature size are dropped.
6. **Relief.** The final vectors are rasterised back to a height field for the 3D preview and STL, so the preview is exactly what the SVGs will cut.

Layer depths: the top layer is the uncut board surface. In **raised** mode the area outside the region is cut to the full depth and the lowest land stands one step above it. In **frame** mode the surroundings stay at full height. The **floor** (default 0.25″) is never cut.

## What changed from the original scripts

The pipeline was rewritten as the `topology` package. It fixes these bugs in the old scripts:

- Pocket bands filled in holes (valleys inside higher ground), so layers overlapped. Now bands are exactly disjoint (tested), and the separate overlap validator is no longer needed.
- Pocket mode ignored the smoothing setting and hard-coded 8 bands regardless of the layer count.
- The SVG was missing padding on the right and top edges.
- Contour points were snapped to whole pixels, which made jagged edges.
- `dem_to_stl` produced real-world-sized meshes (a 250 km state became a 250 km STL). It wasn't watertight, had NaNs outside the boundary, and built faces in a slow Python loop. `svg_to_stl` mixed metre XY with millimetre Z.
- A fixed 30 m DEM for the whole state made large states (California) impossible. The flaky USGS service had no caching of partial failures.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests use a synthetic terrain (no network). They check that bands are disjoint and exactly tile the region, that regions are nested, that holes are preserved, that the STL is watertight, that padding is equal on all sides, and the API job lifecycle.

## Ideas

- Split large maps across multiple glued boards.
- Rivers, lakes and roads as V-carve layers.
- Non-US boundaries by name (countries, national parks).
