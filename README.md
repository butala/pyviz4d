# PyViz4D

A modern, VTK-backed (via PyVista) 4-D spatio-temporal data visualization tool.

## Installation

The core install is deliberately small — `vtk` + `numpy` + `matplotlib`, which is
enough to draw wireframes, voxel cages and points, and to write offscreen PNGs:

```bash
uv pip install pyviz4d
```

```bash
# from a checkout, for development:
uv pip install -e ".[dev]"
```

Everything heavier (Earth textures, CityJSON/CRS, video recording, PhiFlow) is
imported lazily and therefore lives behind extras. Ask for only what you use:

| Extra | Pulls in | Unlocks |
| --- | --- | --- |
| `earth` | `pooch` | `blue_marble.fetch()` → `EarthViewer4D` / `earth_actor` |
| `geo` | `pooch`, `pyproj` | `cityjson.read_cityjson()` (also `examples/demo_cityjson.py`) |
| `video` | `imageio`, `imageio[ffmpeg]` | `Viewer4D.enable_recording()` — `frames_dir` needs plain `imageio`, `video_path` needs ffmpeg |
| `all` | all four leaves | everything except the PhiFlow demo |
| `phiflow` | `phiflow`, `scipy`, `tqdm`, `matplotlib`, `jax` | `examples/demo_phiflow.py` |
| `dev` | `pytest` + the four leaves | running `tests/` |

```bash
uv pip install "pyviz4d[all]"      # or "pyviz4d[earth]", "pyviz4d[geo]", "pyviz4d[video]"
```

`matplotlib` stays in the core set: `primitives.get_color`, `volume.matplotlib_ctf`
and `streamline` import it at module level.

## Running Examples

**Basic Earth Demo:**
```bash
uv run --with vtk --with numpy --with "imageio[ffmpeg]" --with pooch examples/demo_4d.py
```

**Grid Views Demo (Multi-view Linked Camera & Time):**
```bash
uv run --with vtk --with numpy examples/demo_grid.py --nrows 1 --ncols 2
```

**Decoupled Binary Pipeline Demo (Write VTK series & Lazy-load):**
```bash
uv run --with vtk --with numpy --with tqdm --with matplotlib examples/demo_decoupled_pipeline.py
```

**PhiFlow Smoke Plume Demo:**
```bash
uv run --with vtk --with numpy --with phiflow --with scipy --with tqdm --with matplotlib examples/demo_phiflow.py
```

**PlumeViz-style COVIS Hydrothermal Plume Demo (volume rendering + streamlines):**
```bash
uv run --with vtk --with numpy --with scipy --with matplotlib examples/demo_plumeviz.py
```

### Paper-faithful Figure 3 reproduction (best-quality static render)

```bash
uv run --with vtk --with numpy --with scipy --with matplotlib --with pillow \
    examples/demo_plumeviz_fig3.py
```

`examples/demo_plumeviz_fig3.py` reproduces the visual language of the paper's
**Figure 3** — a grey background, a soft golden ray-cast volume, a translucent
blue isosurface envelope, the green plume centreline, and the paper's left-hand
*Plume Height (m)* scale — as a single high-quality static frame. It writes
`examples/plumeviz_fig3.png`.

Why this script exists (and why the older demo looked muddy): COVIS `Id_filt`
is ~99% noise at ~1e-9, and the dense near-seafloor plume is ~1000x brighter
than the faint rising column. The pipeline here handles both ends:

1. **Noise-floor removal** — hard-threshold the volume.
2. **Two-scale smoothing** — a *sharper* Gaussian (`--sigma-display`, 1.0)
   drives the ray-cast volume so internal structure survives, while a
   *smoother* one (`--sigma-iso`, 2.2) drives the marching-cubes envelope and
   the centreline so they come out continuous instead of as blobs.
3. **Connected-component pruning** removes stray floating blobs.
4. **Log scaling** maps the ~1000x backscatter range onto the transfer function.
5. **Framing** on the rising column with a near-side 3/4 camera, on a grey
   background matched to the paper.

The same script renders the paper's other facets via `--mode`:

```bash
# Figure 5 style: dense RK4 velocity streamlines, coloured by speed
uv run --with vtk --with numpy --with scipy --with matplotlib --with pillow \
    examples/demo_plumeviz_fig3.py --mode streamlines \
    --out examples/plumeviz_fig5.png

# Figure 6 style: grey vs rainbow colormap, side by side
uv run --with vtk --with numpy --with scipy --with matplotlib --with pillow \
    examples/demo_plumeviz_fig3.py --mode colormaps \
    --colormap plume_gray --compare-colormap gist_rainbow \
    --size 560 900 --out examples/plumeviz_fig6.png
```

(The Figure-5 velocity field is *synthetic* — the contest does not release the
Doppler data — but it is shaped by the real backscatter, so the streamlines rise
through the actual plume.)

The older interactive demo `examples/demo_plumeviz.py` remains for exploring the
full parameter surface and the other figures:

```bash
# Colormap comparison, Figure 6 style (gray = realistic undersea, rainbow = contrast)
uv run --with vtk --with numpy --with scipy --with matplotlib \
    examples/demo_plumeviz.py --ncols 2 --colormap plume_gray --colormap2 gist_rainbow

# Backscatter-gradient streamlines, Figure 4 style
uv run --with vtk --with numpy --with scipy --with matplotlib \
    examples/demo_plumeviz.py --streamlines

# Synthetic buoyant-velocity streamlines, Figure 5 style
uv run --with vtk --with numpy --with scipy --with matplotlib \
    examples/demo_plumeviz.py --velocity
```

See `examples/plumeviz_preview.png` for a rendered frame, and
`examples/demo_plumeviz.py` for the full parameter surface (colormap, opacity
transfer function, isovalue, segmentation threshold, dilation, streamlines).
