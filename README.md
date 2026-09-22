# PyViz4D

Scientific and spatio-temporal (4-D) visualization built directly on **VTK** —
no PyVista. Time-varying actors, volume rendering, streamlines, CityJSON city
models, Earth textures, and a small palette of primitives for drawing lines,
points and spherical voxels.

## Installation

The core install is deliberately small — `vtk` + `numpy` + `matplotlib`, enough
to draw wireframes, voxel cages and points and to write offscreen PNGs:

```bash
uv pip install pyviz4d
```

Everything heavier (Earth textures, CityJSON/CRS, video recording, PhiFlow) is
imported lazily and therefore lives behind extras. Ask for only what you use:

| Extra | Pulls in | Unlocks |
| --- | --- | --- |
| `earth` | `pooch` | `pyviz4d.earth` (textures, `WGS84`) and `EarthViewer4D` |
| `geo` | `pooch`, `pyproj` | `read_cityjson()`, `examples/demo_cityjson.py` |
| `video` | `imageio`, `imageio[ffmpeg]` | `Viewer4D.enable_recording()` — `frames_dir` needs plain `imageio`, `video_path` needs ffmpeg |
| `all` | all four leaves | everything except the PhiFlow demo |
| `phiflow` | `phiflow`, `scipy`, `tqdm`, `matplotlib`, `jax` | `examples/demo_phiflow.py` |
| `dev` | `pytest` + the four leaves | running `tests/` |

```bash
uv pip install "pyviz4d[all]"      # or [earth], [geo], [video]
```

`matplotlib` stays in the core set because `primitives.get_color`,
`volume.matplotlib_ctf` and `streamline` import it at module level.

## Quick start

```python
import numpy as np
from pyviz4d import Viewer4D, line_actor, point_actor, render_to_png

xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
wire = line_actor(xyz, xyz + 1.0)          # one cell per pair, cyan by default
dots = point_actor(xyz, size=8.0)          # `size` is a glyph radius in world units

render_to_png([wire, dots], "scene.png", size=(1200, 900))   # offscreen, no display

viewer = Viewer4D(size=(1200, 900))
viewer.add_actor(wire)
viewer.add_actor(dots)
viewer.save_screenshot("shot.png", scale=2)  # works without start(); scale=2 doubles pixels
viewer.start()                             # blocks in the interactor loop
```

`render_to_png` creates its own renderer and offscreen window, so a one-shot
PNG needs no interactor. Note that it ends with `ResetCamera()`, which reframes
the supplied `camera=` to fit the scene and cancels any pre-applied `Zoom()` —
only the view direction, focal point and view-up survive, so aim the camera
rather than positioning it at a chosen radius.

## Coordinate convention

Every angular helper uses **latitude**, not colatitude:

```
x = r cos(theta) cos(phi)     y = r cos(theta) sin(phi)     z = r sin(theta)
```

with `theta` in `[-pi/2, +pi/2]`. This matches `pyviz4d.earth` (WGS84
conversions) and `pyviz4d.spherical_grid`, which is used by
`examples/demo_4d.py`. There is intentionally **no colatitude helper** anywhere
in the package: the same name with the opposite convention is the failure mode
this codebase is written to avoid.

## API at a glance

| Module | Contents |
| --- | --- |
| `pyviz4d.primitives` | `get_color`, `line_source`, `line_actor`, `point_actor`, `spherical_voxel_actor`, `render_to_png` |
| `pyviz4d.viz` | `Viewer4D` (`add_actor`, `start`, `enable_recording`, `save_screenshot`), `EarthViewer4D`, `TemporalActor` |
| `pyviz4d.volume` | `VolumeActor`, `IsosurfaceActor`, `matplotlib_ctf`, `power_opacity`, `create_vtk_image_from_numpy` |
| `pyviz4d.streamline` | `StreamlineActor`, `CenterlineActor`, `trace_streamlines`, `gradient_field`, `extract_centerline` |
| `pyviz4d.series` / `pyviz4d.io` | `IsosurfaceSeriesActor`, `PolyDataSeriesActor`, `VTKSeriesWriter`, `parse_pvd` |
| `pyviz4d.cityjson` | `read_cityjson` |
| `pyviz4d.earth` | `earth_actor`, `WGS84`, `blue_marble.fetch` |

## Examples

```bash
uv sync --extra all          # once; or --extra dev for the test suite
```

| Example | What it shows |
| --- | --- |
| `demo_4d.py` | Earth texture with a time animation |
| `demo_grid.py` | Multi-view grid with linked camera and time (`--nrows`, `--ncols`) |
| `demo_cityjson.py` | Reading a CityJSON city model |
| `demo_decoupled_pipeline.py` | Writing a VTK time series, then lazy-loading it |
| `demo_phiflow.py` | PhiFlow smoke plume |
| `demo_plumeviz.py` | COVIS hydrothermal plume: volume rendering + streamlines |
| `demo_plumeviz_fig3.py` | Paper-faithful static figures (`--mode`) |
| `validate_wgs84.py` | WGS84 conversion checks |
| `demo_lod1_*.py` | LoD1 city models from OpenStreetMap (below) |

```bash
uv run python examples/demo_grid.py --nrows 1 --ncols 2
uv run python examples/demo_cityjson.py
```

### Paper-faithful PlumeViz figures

`examples/demo_plumeviz_fig3.py` reproduces the visual language of the paper's
Figure 3 — grey background, soft golden ray-cast volume, translucent blue
isosurface envelope, green plume centreline, left-hand *Plume Height (m)* scale
— as one high-quality static frame. It writes `examples/plumeviz_fig3.png`.

The same script renders the other facets via `--mode`. The COVIS `Id_filt`
volume is ~99% noise near ~1e-9 and the near-seafloor plume is ~1000x brighter
than the faint rising column, so the pipeline hard-thresholds the noise floor,
drives the ray-cast volume with a *sharper* Gaussian (`--sigma-display`) and the
marching-cubes envelope and centreline with a *smoother* one (`--sigma-iso`),
prunes stray connected components, log-scales the ~1000x backscatter range onto
the transfer function, and frames the rising column from a near-side 3/4 camera.

```bash
# Figure 5 style: RK4 velocity streamlines, coloured by speed
uv run python examples/demo_plumeviz_fig3.py --mode streamlines \
    --out examples/plumeviz_fig5.png

# Figure 6 style: grey vs rainbow colormap, side by side
uv run python examples/demo_plumeviz_fig3.py --mode colormaps \
    --colormap plume_gray --compare-colormap gist_rainbow \
    --size 560 900 --out examples/plumeviz_fig6.png
```

The Figure-5 velocity field is *synthetic* — the contest does not release the
Doppler data — but it is shaped by the real backscatter, so the streamlines rise
through the actual plume. `examples/demo_plumeviz.py` remains the interactive
demo over the full parameter surface (colormap, opacity transfer function,
isovalue, segmentation threshold, dilation, streamlines):

```bash
uv run python examples/demo_plumeviz.py --ncols 2 \
    --colormap plume_gray --colormap2 gist_rainbow      # Figure 6 style
uv run python examples/demo_plumeviz.py --streamlines    # Figure 4 style
uv run python examples/demo_plumeviz.py --velocity       # Figure 5 style
```

### LoD1 city models from OpenStreetMap

Three worked examples geocode a place, pull OSM footprints, extrude LoD1 solids
and write a CityJSON 1.1 model plus a PNG:

```bash
uv run python examples/demo_lod1_lian.py      # Li'an Education Zone, Hainan
uv run python examples/demo_lod1_pudong.py    # Lujiazui, Shanghai (real 632 m heights)
uv run python examples/demo_lod1_glasgow.py   # James Watt Building, Glasgow
uv run python examples/demo_lod1_view.py      # interactive viewer for any of them
```

Heights come from OSM `height` / `building:levels` where tagged, otherwise a
per-type default; each run prints the `height source` breakdown, so check it
before trusting the vertical dimension. Outputs land in `data/` (gitignored), so
running the examples never dirties the tree. Building data is
&copy; OpenStreetMap contributors, [ODbL 1.0](https://www.openstreetmap.org/copyright).

## Tests

```bash
uv sync --extra dev
uv run pytest tests/ -q
```

One test cross-checks the ported primitives against SphericalCT's reference
implementation and skips itself when that checkout is not present on the machine.

## Data

No third-party datasets are redistributed here. The LoD1 examples fetch
OpenStreetMap data at runtime; the PlumeViz examples expect COVIS contest data
to be present locally.
