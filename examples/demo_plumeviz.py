"""
PyViz4D demo reproducing the visual language of PlumeViz (2024 IEEE SciVis
Contest winner) for COVIS hydrothermal-plume sonar data.

    Figure 3 -> volume rendering (yellow/orange, log-scaled opacity) + a blue
                isosurface + the green plume centreline.
    Figure 4 -> 4th-order Runge-Kutta streamlines of the backscatter gradient,
                seeded along the centreline and flowing centre -> periphery.
    Figure 5 -> streamlines of a (synthetic, buoyant-plume) velocity field.
    Figure 6 -> the same backscatter volume with two colormaps side by side.

The real contest data are MATLAB ``*-imaging1.mat`` files (see the contest page:
https://sciviscontest2024.github.io/data/). Any downloaded files placed in
``data/covis/`` are picked up automatically; if none are present the demo falls
back to a synthetic plume so you can still try it.

Usage (interactive):
    uv run --with vtk --with numpy --with scipy --with matplotlib examples/demo_plumeviz.py

Colormap comparison (Figure 6 style, two linked views):
    uv run --with vtk --with numpy --with scipy --with matplotlib \\
        examples/demo_plumeviz.py --ncols 2 --colormap plume_gray --colormap2 gist_rainbow

Offline single-frame render (great for tweaking the look):
    uv run --with vtk --with numpy --with scipy --with matplotlib \\
        examples/demo_plumeviz.py --screenshot plume.png --frame 1
"""

import argparse
import glob
import os

import numpy as np
import vtk

from pyviz4d import (Viewer4D, VolumeActor, IsosurfaceActor, StreamlineActor,
                     CenterlineActor, matplotlib_ctf, power_opacity,
                     extract_centerline, gradient_field)


def _register_plume_colormaps():
    """Register a couple of vivid, paper-flavoured colormaps so they can be
    used by name in ``--colormap``."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    if "plume_gold" not in matplotlib.colormaps():
        gold = LinearSegmentedColormap.from_list("plume_gold", [
            (0.00, (0.24, 0.09, 0.02)),
            (0.30, (0.52, 0.21, 0.00)),
            (0.55, (0.83, 0.40, 0.02)),
            (0.75, (1.00, 0.62, 0.08)),
            (0.90, (1.00, 0.84, 0.40)),
            (1.00, (1.00, 0.98, 0.84)),
        ])
        matplotlib.colormaps.register(gold)
    if "plume_gray" not in matplotlib.colormaps():
        gray = LinearSegmentedColormap.from_list("plume_gray", [
            (0.00, (0.03, 0.03, 0.04)),
            (0.35, (0.28, 0.28, 0.30)),
            (0.65, (0.55, 0.55, 0.58)),
            (0.85, (0.80, 0.80, 0.82)),
            (1.00, (0.97, 0.97, 0.98)),
        ])
        matplotlib.colormaps.register(gray)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_covis(path, roi=None):
    """Load one COVIS imaging-mode .mat file into an Id_filt volume.

    Returns ``(volume, spacing, origin, name)``. ``volume`` has shape (X, Y, Z).
    """
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    key = [k for k in m if not k.startswith("__")][0]
    grid = m[key].grid

    vol = np.asarray(grid.Id_filt, dtype=np.float32)  # differenced + OS-CFAR filtered
    axis = np.asarray(grid.axis).ravel().astype(float)
    size = np.asarray(grid.size).ravel().astype(int)

    spacing = (float(grid.spacing.dx), float(grid.spacing.dy), float(grid.spacing.dz))
    origin = (float(axis[0]), float(axis[2]), float(axis[4]))

    if roi is not None:
        x0, x1, y0, y1, z0, z1 = roi
        i0 = int(np.clip(round((x0 - origin[0]) / spacing[0]), 0, size[0] - 1))
        i1 = int(np.clip(round((x1 - origin[0]) / spacing[0]), 0, size[0] - 1))
        j0 = int(np.clip(round((y0 - origin[1]) / spacing[1]), 0, size[1] - 1))
        j1 = int(np.clip(round((y1 - origin[1]) / spacing[1]), 0, size[1] - 1))
        k0 = int(np.clip(round((z0 - origin[2]) / spacing[2]), 0, size[2] - 1))
        k1 = int(np.clip(round((z1 - origin[2]) / spacing[2]), 0, size[2] - 1))
        vol = vol[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
        origin = (origin[0] + i0 * spacing[0],
                  origin[1] + j0 * spacing[1],
                  origin[2] + k0 * spacing[2])

    name = os.path.basename(path)
    return vol, spacing, origin, name


def synthetic_plume(res=(96, 96, 64), frames=4):
    """Fallback synthetic buoyant plume (bent, widening with height)."""
    from scipy.ndimage import gaussian_filter
    xs = np.linspace(-20, 20, res[0])
    ys = np.linspace(-20, 20, res[1])
    zs = np.linspace(0, 16, res[2])
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    volumes = []
    for f in range(frames):
        phase = f * 0.6
        bend = 0.35 * Z + 1.2 * np.sin(phase)
        cx = 2.0 + bend
        cy = -1.0 + 0.5 * np.sin(0.5 * Z + phase)
        sigma = 1.2 + 0.16 * Z
        r2 = ((X - cx) ** 2 + (Y - cy) ** 2) / (sigma ** 2)
        amp = np.exp(-0.5 * r2) * np.exp(-0.25 * Z) * (0.4 + 0.6 * np.exp(-0.15 * Z))
        amp += 0.02 * np.random.default_rng(f).normal(size=X.shape)
        amp = gaussian_filter(amp.astype(np.float32), 1.0)
        volumes.append(np.clip(amp, 0, None).astype(np.float32))

    spacing = (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0])
    origin = (xs[0], ys[0], zs[0])
    return volumes, spacing, origin


# --------------------------------------------------------------------------- #
# Scene construction
# --------------------------------------------------------------------------- #

def log_scale(vol, vmin):
    """Log-transform backscatter so low values stay transparent (as in PlumeViz)."""
    v = np.clip(vol, vmin, None)
    return np.log10(v)


def smooth(vol, sigma=1.0):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(vol.astype(np.float32), sigma)


def segment_plume(vol, threshold):
    """Hard-threshold the volume to drop the sonar noise floor.

    COVIS backscatter is extremely sparse: ~99% of voxels sit at ~1e-9 while
    the plume lives around 1e-4..0.9. Zeroing everything below ``threshold``
    keeps the rendered volume crisp instead of a fog of dim noise.
    """
    return np.where(vol >= threshold, vol, 0.0).astype(np.float32)


def fatten(vol, size=3, sigma=0.7):
    """Gently dilate a sparse sonar plume so direct volume rendering reads as a
    solid glowing structure rather than faint scattered points. This is a visual
    aid applied *only* to the rendered scalar field; analysis (centreline,
    streamlines, isosurface) still uses the original values."""
    from scipy.ndimage import maximum_filter, gaussian_filter
    v = maximum_filter(vol, size=size).astype(np.float32)
    if sigma > 0:
        v = gaussian_filter(v, sigma)
    return v


def buoyant_velocity(vol, spacing, origin, rise=1.0, entrain=0.15):
    """Synthetic buoyant-plume velocity field (Figure 5 stand-in for Doppler).

    Rises where backscatter is strong and entrains ambient fluid toward the
    centreline, mimicking a turbulent buoyant plume.
    """
    nx, ny, nz = vol.shape
    pts, _ = extract_centerline(vol, spacing, origin)

    xc = np.full(nz, np.nan)
    yc = np.full(nz, np.nan)
    for p in pts:
        k = int(np.clip(round((p[2] - origin[2]) / spacing[2]), 0, nz - 1))
        xc[k], yc[k] = p[0], p[1]

    def fill(a):
        idx = np.arange(nz)
        mask = ~np.isnan(a)
        return np.interp(idx, idx[mask], a[mask]) if mask.any() else np.zeros(nz)

    xc, yc = fill(xc), fill(yc)

    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]
    zs = origin[2] + np.arange(nz) * spacing[2]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    vmax = vol.max() if vol.max() > 0 else 1.0
    strength = np.clip(vol / vmax, 0.0, 1.0)

    vx = -entrain * (X - xc[None, None, :]) * strength
    vy = -entrain * (Y - yc[None, None, :]) * strength
    vz = rise * strength
    return vx.astype(np.float32), vy.astype(np.float32), vz.astype(np.float32)


def build_scene(volumes, spacing, origin, args):
    """Build all actors. Returns ``(actors, cam_bounds, ctf, scalar_bar)``."""
    seg_thr = args.segment_threshold
    n_frames = len(volumes)

    # 1. Segment (drop noise) + fatten (fill the sparse plume) for display. --
    render_volumes = []
    for v in volumes:
        sv = segment_plume(v, seg_thr)
        if args.dilate > 0:
            sv = fatten(sv, size=args.dilate, sigma=args.dilate_sigma)
        render_volumes.append(sv)

    # Log-scale the (segmented) data: spreads the huge backscatter range and
    # lets us allocate colour/opacity to the plume rather than to noise.
    frames_log = [log_scale(v, seg_thr) for v in render_volumes]

    log_min = np.log10(seg_thr)
    log_max = float(np.log10(max(v.max() for v in volumes)))

    actors = []

    # 1. Volume rendering ----------------------------------------------------
    ctf = matplotlib_ctf(args.colormap, log_min, log_max, n=128)
    pwf = power_opacity(log_min, log_max, power=args.opacity_power,
                        max_opacity=args.max_opacity, n=128)

    vol_actor = VolumeActor(frames_log, spacing=spacing, origin=origin)
    vol_actor.prop.SetColor(ctf)
    vol_actor.prop.SetScalarOpacity(pwf)
    vol_actor.prop.SetShade(1)
    vol_actor.prop.SetInterpolationTypeToLinear()
    vol_actor.prop.SetAmbient(0.35)
    vol_actor.prop.SetDiffuse(0.9)
    vol_actor.prop.SetSpecular(0.3)
    vol_actor.prop.SetSpecularPower(40)
    vol_actor.mapper.SetBlendModeToComposite()
    vol_actor.mapper.SetSampleDistance(spacing[0] * 0.4)
    actors.append(vol_actor)

    ctf2 = None
    if args.ncols == 2:
        # A second volume actor with a different colormap for the right view.
        ctf2 = matplotlib_ctf(args.colormap2, log_min, log_max, n=128)
        vol2 = VolumeActor(frames_log, spacing=spacing, origin=origin)
        vol2.prop.SetColor(ctf2)
        vol2.prop.SetScalarOpacity(pwf)
        vol2.prop.SetShade(1)
        vol2.prop.SetAmbient(0.35)
        vol2.prop.SetDiffuse(0.9)
        vol2.prop.SetSpecular(0.3)
        vol2.prop.SetSpecularPower(40)
        vol2.mapper.SetBlendModeToComposite()
        vol2.mapper.SetSampleDistance(spacing[0] * 0.4)
        actors.append(vol2)

    # 2. Blue isosurface (Figure 3) ------------------------------------------
    # Extract from the same fattened field used for the volume render, so the
    # sparse sonar points fuse into a clean, closed envelope.
    if not args.no_isosurface:
        iso = IsosurfaceActor(
            render_volumes, spacing=spacing, origin=origin,
            iso_values=[args.iso_value],
            colors=[(0.16, 0.42, 0.95)],
            opacity=args.iso_opacity)
        actors.append(iso)

    # 3. Green centreline (Figure 3) -----------------------------------------
    if not args.no_centerline:
        centerline = CenterlineActor(
            volumes, spacing=spacing, origin=origin,
            threshold=args.centerline_threshold,
            color=(0.05, 0.85, 0.2),
            tube_radius=args.centerline_radius)
        actors.append(centerline)

    # 4. Streamlines ---------------------------------------------------------
    if args.streamlines or args.velocity:
        if args.velocity:
            frames_vec = [buoyant_velocity(v, spacing, origin,
                                           rise=args.rise, entrain=args.entrain)
                          for v in volumes]
            seeds = velocity_seeds(volumes[0], spacing, origin, n=args.n_seeds)
            direction = "forward"
            colormap = args.streamline_colormap
            magnitude_range = None
        else:
            frames_vec = []
            for v in volumes:
                gx, gy, gz = gradient_field(smooth(v, args.smooth_sigma), spacing)
                # Negative gradient -> flow from centre outward (Figure 4).
                frames_vec.append((-gx, -gy, -gz))
            seeds = gradient_seeds(volumes[0], spacing, origin,
                                   threshold=args.centerline_threshold,
                                   jitter=args.seed_jitter,
                                   n_radial=args.n_radial)
            direction = args.streamline_direction
            colormap = args.streamline_colormap
            magnitude_range = None

        streamlines = StreamlineActor(
            frames_vec, spacing=spacing, origin=origin, seeds=seeds,
            direction=direction,
            max_propagation=args.max_propagation,
            initial_step=args.initial_step,
            color_by_magnitude=True, colormap=colormap,
            line_width=args.streamline_width)
        actors.append(streamlines)

    # 5. Bounding box / height scale -----------------------------------------
    bounds = compute_bounds(origin, spacing, volumes[0].shape)
    cube_axes = make_cube_axes(bounds)
    actors.append(cube_axes)

    # 6. Scalar bar (the colour scale the paper draws beside its figures) ----
    scalar_bar = make_scalar_bar(ctf, log_min, log_max, args.colormap)

    return actors, bounds, ctf, scalar_bar


def compute_bounds(origin, spacing, shape):
    nx, ny, nz = shape
    return (origin[0], origin[0] + (nx - 1) * spacing[0],
            origin[1], origin[1] + (ny - 1) * spacing[1],
            origin[2], origin[2] + (nz - 1) * spacing[2])


def make_cube_axes(bounds):
    cube = vtk.vtkCubeAxesActor()
    cube.SetBounds(*bounds)
    cube.SetFlyModeToOuterEdges()
    cube.SetXTitle("Easting (m)")
    cube.SetYTitle("Northing (m)")
    cube.SetZTitle("Height (m)")
    for i in range(3):
        cube.GetTitleTextProperty(i).SetColor(0.9, 0.9, 0.9)
        cube.GetLabelTextProperty(i).SetColor(0.7, 0.7, 0.7)
        cube.GetLabelTextProperty(i).SetFontSize(14)
    cube.GetProperty().SetColor(0.35, 0.35, 0.35)
    return cube


def make_scalar_bar(ctf, scalar_min, scalar_max, cmap_name):
    """Vertical colour-scale annotation beside the volume (like the paper)."""
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(ctf)
    bar.SetTitle(f"{cmap_name}  log10 backscatter")
    bar.SetNumberOfLabels(5)
    bar.SetWidth(0.08)
    bar.SetHeight(0.6)
    bar.SetPosition(0.02, 0.2)
    bar.GetTitleTextProperty().SetColor(0.9, 0.9, 0.9)
    bar.GetTitleTextProperty().SetFontSize(12)
    bar.GetLabelTextProperty().SetColor(0.75, 0.75, 0.75)
    bar.GetLabelTextProperty().SetFontSize(11)
    bar.SetLabelFormat("%.1f")
    return bar


def gradient_seeds(vol, spacing, origin, threshold=None, jitter=0.3, n_radial=8):
    from pyviz4d.streamline import centerline_seeds
    seeds = centerline_seeds(vol, spacing, origin, threshold=threshold,
                             radial_jitter=jitter, n_radial=n_radial, seed=0)
    # Subsample along z to avoid an over-crowded field.
    n = seeds.GetNumberOfPoints()
    keep = vtk.vtkPoints()
    for i in range(0, n, 2):
        keep.InsertNextPoint(seeds.GetPoint(i))
    out = vtk.vtkPolyData()
    out.SetPoints(keep)
    return out


def velocity_seeds(vol, spacing, origin, n=24):
    pts, _ = extract_centerline(vol, spacing, origin)
    if len(pts) == 0:
        pts = np.array([[origin[0], origin[1], origin[2]]])
    base = pts[np.argmin(np.abs(pts[:, 2] - origin[2]))]
    seeds = vtk.vtkPoints()
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        for r in (2.0, 4.0):
            seeds.InsertNextPoint(base[0] + r * np.cos(a),
                                  base[1] + r * np.sin(a),
                                  base[2] + 0.5)
    out = vtk.vtkPolyData()
    out.SetPoints(seeds)
    return out


# --------------------------------------------------------------------------- #
# Camera & rendering helpers
# --------------------------------------------------------------------------- #

def position_camera(renderer, bounds):
    x0, x1, y0, y1, z0, z1 = bounds
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    cam = renderer.GetActiveCamera()
    # 3/4 front view, mildly elevated so the rising plume fills the frame.
    cam.SetPosition(cx + 1.4 * dx, cy + 1.5 * dy, cz + 0.7 * dz)
    cam.SetFocalPoint(cx, cy, 0.35 * dz + z0)
    cam.SetViewUp(0, 0, 1)
    renderer.ResetCameraClippingRange()


def set_cube_axes_camera(actors, renderer):
    cam = renderer.GetActiveCamera()
    for a in actors:
        if isinstance(a, vtk.vtkCubeAxesActor):
            a.SetCamera(cam)


def offscreen_render(actors, bounds, size, bg, path, frame_idx, scalar_bar=None):
    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.SetSize(*size)
    ren = vtk.vtkRenderer()
    ren.SetBackground(*bg)
    ren.SetUseDepthPeeling(1)
    ren.SetOcclusionRatio(0.1)
    ren.SetMaximumNumberOfPeels(8)
    ren_win.SetAlphaBitPlanes(1)
    ren_win.SetMultiSamples(8)
    ren_win.AddRenderer(ren)

    for a in actors:
        if hasattr(a, "update"):
            a.update(float(frame_idx))
        vtk_actor = a.actor if hasattr(a, "actor") else a
        ren.AddActor(vtk_actor)

    if scalar_bar is not None:
        ren.AddViewProp(scalar_bar)

    position_camera(ren, bounds)
    set_cube_axes_camera(actors, ren)
    ren_win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(ren_win)
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"Saved screenshot to {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser(description="PlumeViz-style COVIS plume visualization")
    p.add_argument("--data-dir", default="data/covis")
    p.add_argument("--roi", nargs=6, type=float, default=[-2, 20, -40, -16, 0, 16],
                   metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                   help="region of interest in metres (x0 x1 y0 y1 z0 z1)")
    p.add_argument("--ncols", type=int, default=1, choices=[1, 2])
    p.add_argument("--colormap", default="YlOrBr_r", help="Matplotlib colormap for volume (paper's yellow; 'plume_gold', 'plume_gray', 'gist_rainbow' also work)")
    p.add_argument("--colormap2", default="gist_rainbow", help="second colormap when --ncols 2")
    p.add_argument("--segment-threshold", type=float, default=1e-5,
                   help="hard threshold below which backscatter is treated as noise")
    p.add_argument("--opacity-power", type=float, default=1.4,
                   help="opacity ramp exponent in log space (1 = linear; >1 keeps the plume edge translucent)")
    p.add_argument("--max-opacity", type=float, default=0.6)
    p.add_argument("--dilate", type=int, default=7, help="grey-dilation size for the rendered volume (0 disables)")
    p.add_argument("--dilate-sigma", type=float, default=1.0, help="post-dilation Gaussian sigma")

    p.add_argument("--iso-value", type=float, default=0.0005)
    p.add_argument("--iso-opacity", type=float, default=0.7)
    p.add_argument("--no-isosurface", action="store_true")

    p.add_argument("--centerline-threshold", type=float, default=1e-4)
    p.add_argument("--centerline-radius", type=float, default=0.28)
    p.add_argument("--no-centerline", action="store_true")

    p.add_argument("--streamlines", action="store_true",
                   help="add backscatter-gradient streamlines (Fig. 4); off by default for a clean Fig. 3")
    p.add_argument("--velocity", action="store_true",
                   help="trace synthetic buoyant-velocity streamlines (Fig. 5) instead of gradient streamlines")
    p.add_argument("--streamline-colormap", default="plasma")
    p.add_argument("--streamline-direction", default="forward", choices=["forward", "backward", "both"])
    p.add_argument("--streamline-width", type=float, default=2.5)
    p.add_argument("--n-seeds", type=int, default=24)
    p.add_argument("--n-radial", type=int, default=6)
    p.add_argument("--seed-jitter", type=float, default=0.35)
    p.add_argument("--smooth-sigma", type=float, default=1.0)
    p.add_argument("--max-propagation", type=float, default=12.0)
    p.add_argument("--initial-step", type=float, default=0.08)
    p.add_argument("--rise", type=float, default=1.2)
    p.add_argument("--entrain", type=float, default=0.18)

    p.add_argument("--screenshot", default=None, help="render a single frame to a PNG and exit")
    p.add_argument("--frame", type=int, default=1, help="frame index for --screenshot")
    p.add_argument("--interp-steps", type=int, default=0,
                   help="linearly interpolate N extra frames between loaded timesteps")
    args = p.parse_args()

    _register_plume_colormaps()

    # Load data ----------------------------------------------------------------
    files = sorted(glob.glob(os.path.join(args.data_dir, "*.mat")))
    volumes = []
    if files:
        for f in files:
            vol, spacing, origin, name = load_covis(f, roi=args.roi)
            volumes.append(vol)
        print(f"Loaded {len(files)} COVIS frames from {args.data_dir}")
    else:
        print("No .mat files found; using a synthetic buoyant plume.")
        volumes, spacing, origin = synthetic_plume()
        args.roi = None

    if args.interp_steps > 0 and len(volumes) > 1:
        interp = []
        for i in range(len(volumes) - 1):
            for s in range(args.interp_steps + 1):
                t = s / (args.interp_steps + 1)
                interp.append((1 - t) * volumes[i] + t * volumes[i + 1])
        interp.append(volumes[-1])
        volumes = interp
        print(f"Temporal interpolation -> {len(volumes)} frames")

    actors, bounds, ctf, scalar_bar = build_scene(volumes, spacing, origin, args)

    if args.screenshot:
        offscreen_render(actors, bounds, (1280, 860), (0.05, 0.05, 0.07),
                         args.screenshot, args.frame, scalar_bar)
        return

    # Interactive viewer -------------------------------------------------------
    viewer = Viewer4D(size=(1280, 860), bg_color=(0.05, 0.05, 0.07),
                      nrows=1, ncols=args.ncols)
    viewer.ren.SetUseDepthPeeling(1)
    viewer.ren.SetOcclusionRatio(0.1)
    viewer.ren.SetMaximumNumberOfPeels(8)
    viewer.ren_win.SetAlphaBitPlanes(1)
    viewer.ren_win.SetMultiSamples(8)

    if args.ncols == 2:
        vol_a, vol_b, *rest = actors
        viewer.add_actor(vol_a, view=0)
        viewer.add_actor(vol_b, view=1)
        for a in rest:
            viewer.add_actor(a, view=0)
        viewer.ren.AddViewProp(scalar_bar)
    else:
        for a in actors:
            viewer.add_actor(a)
        viewer.ren.AddViewProp(scalar_bar)

    position_camera(viewer.ren, bounds)
    set_cube_axes_camera(actors, viewer.ren)
    viewer.add_playback_ui(max_time=len(volumes) - 1, loop=True)
    print("Launching PyViz4D PlumeViz-style viewer...")
    viewer.start(timer_interval_ms=16)


if __name__ == "__main__":
    main()
