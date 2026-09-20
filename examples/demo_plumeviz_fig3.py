"""
Reproduce the visual language of **Figure 3** of the 2024 IEEE SciVis Contest
winner *PlumeViz* using PyViz4D + VTK, from the real COVIS sonar data:

    https://sciviscontest2024.github.io/data/

Figure 3 of the paper shows a hydrothermal plume rendered with a *combination of
direct (ray-casting) and indirect (marching cubes) volume rendering*:

    * a soft, golden **volume** whose opacity grows with backscatter intensity
      (higher values -> more opaque), on a **grey** background,
    * a translucent **blue isosurface** marking a single value of interest,
    * the **green plume centreline** (per-slice maximum backscatter, smoothed),
    * a left-hand **"Plume Height (m)"** scale.

How this script closes in on that look
--------------------------------------
1. **Noise floor removal.** COVIS ``Id_filt`` is extremely sparse (~99% of
   voxels sit at ~1e-9). We hard-threshold to drop the sonar noise floor.
2. **Two-scale smoothing.** Two Gaussian-smoothed copies of the segmented field:
   a *sharper* one (``--sigma-display``) drives the volume so internal structure
   survives, and a *smoother* one (``--sigma-iso``) drives the marching-cubes
   envelope and the centreline so they are continuous and clean.
3. **Connected-component pruning** removes stray floating blobs.
4. **Log scaling** maps the ~1000x backscatter range onto the transfer function.
5. **Framing** on the rising column, a near-side 3/4 camera, and a light grey
   background matched to the paper.

Usage
-----
    uv run --with vtk --with numpy --with scipy --with matplotlib --with pillow \\
        examples/demo_plumeviz_fig3.py

    # a different time step / region
    uv run --with vtk --with numpy --with scipy --with matplotlib --with pillow \\
        examples/demo_plumeviz_fig3.py --frame 3 --roi -4 8 -20 -8 0 11
"""

import argparse
import glob
import os

import numpy as np
import vtk
from scipy.ndimage import gaussian_filter, label

from pyviz4d import create_vtk_image_from_numpy, extract_centerline, matplotlib_ctf
from pyviz4d.volume import power_opacity


# --------------------------------------------------------------------------- #
# Colormap
# ---------------------------------------------------------------------------

def register_plume_colormaps():
    """Register the paper's warm gold ramp (dark amber -> gold -> cream)."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    if "plume_gold" not in matplotlib.colormaps():
        matplotlib.colormaps.register(LinearSegmentedColormap.from_list("plume_gold", [
            (0.00, (0.70, 0.52, 0.20)),
            (0.50, (0.99, 0.84, 0.42)),
            (1.00, (1.00, 0.99, 0.90)),
        ]))
    if "plume_gray" not in matplotlib.colormaps():
        matplotlib.colormaps.register(LinearSegmentedColormap.from_list("plume_gray", [
            (0.00, (0.08, 0.08, 0.10)),
            (0.40, (0.42, 0.42, 0.44)),
            (0.70, (0.72, 0.72, 0.73)),
            (1.00, (0.98, 0.98, 0.98)),
        ]))


# --------------------------------------------------------------------------- #
# Data
# ---------------------------------------------------------------------------

def load_covis(path, roi):
    """Load a COVIS ``*-imaging1.mat`` file and crop it to ``roi`` (metres)."""
    import scipy.io as sio
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    key = [k for k in m if not k.startswith("__")][0]
    grid = m[key].grid

    vol = np.asarray(grid.Id_filt, dtype=np.float32)             # (X, Y, Z)
    axis = np.asarray(grid.axis).ravel().astype(float)           # [x0 x1 y0 y1 z0 z1]
    spacing = (float(grid.spacing.dx), float(grid.spacing.dy), float(grid.spacing.dz))
    origin = (float(axis[0]), float(axis[2]), float(axis[4]))

    if roi is not None:
        x0, x1, y0, y1, z0, z1 = roi
        i = lambda a, b, o, s, n: (int(np.clip(round((a - o) / s), 0, n - 1)),
                                   int(np.clip(round((b - o) / s), 0, n - 1)))
        i0, i1 = i(x0, x1, origin[0], spacing[0], vol.shape[0])
        j0, j1 = i(y0, y1, origin[1], spacing[1], vol.shape[1])
        k0, k1 = i(z0, z1, origin[2], spacing[2], vol.shape[2])
        vol = vol[i0:i1 + 1, j0:j1 + 1, k0:k1 + 1]
        origin = (origin[0] + i0 * spacing[0],
                  origin[1] + j0 * spacing[1],
                  origin[2] + k0 * spacing[2])

    return vol, spacing, origin, os.path.basename(path)


# --------------------------------------------------------------------------- #
# Pre-processing
# ---------------------------------------------------------------------------

def prepare(vol, seg_thr, sigma_iso, sigma_display, prune_frac):
    """Return ``(log_display, sm_iso, sm_display)``.

    ``log_display`` feeds the ray-cast volume, ``sm_iso`` the marching-cubes
    envelope + centreline.
    """
    seg = np.where(vol >= seg_thr, vol, 0.0).astype(np.float32)
    sm_iso = gaussian_filter(seg, sigma_iso).astype(np.float32)
    sm_disp = gaussian_filter(seg, sigma_display).astype(np.float32)

    if prune_frac > 0:
        lab, n = label(sm_iso > 0.02 * float(sm_iso.max()))
        if n > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            keep = np.zeros(n + 1, bool)
            keep[sizes >= prune_frac * sizes.max()] = True
            mask = keep[lab]
            sm_iso = np.where(mask, sm_iso, 0.0).astype(np.float32)
            sm_disp = np.where(mask, sm_disp, 0.0).astype(np.float32)

    log_display = np.log10(np.clip(sm_disp, seg_thr, None)).astype(np.float32)
    return log_display, sm_iso, sm_disp


# --------------------------------------------------------------------------- #
# Scene
# ---------------------------------------------------------------------------

def build_scene(log_display, sm_iso, spacing, origin, shape, args):
    lo, hi = float(log_display.min()), float(log_display.max())
    actors = []

    # 1. Direct volume rendering (ray casting) --------------------------------
    image = create_vtk_image_from_numpy(log_display, spacing, origin)
    mapper = vtk.vtkSmartVolumeMapper()
    mapper.SetInputData(image)
    mapper.SetBlendModeToComposite()
    mapper.SetAutoAdjustSampleDistances(0)
    mapper.SetSampleDistance(spacing[0] * 0.35)

    prop = vtk.vtkVolumeProperty()
    prop.SetColor(matplotlib_ctf(args.colormap, lo, hi, n=128))
    prop.SetScalarOpacity(power_opacity(lo, hi, power=args.opacity_power,
                                        max_opacity=args.max_opacity, n=128))
    prop.ShadeOn()
    prop.SetInterpolationTypeToLinear()
    prop.SetAmbient(0.45)
    prop.SetDiffuse(0.85)
    prop.SetSpecular(0.35)
    prop.SetSpecularPower(30)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volume.SetProperty(prop)
    actors.append(volume)

    # 2. Indirect volume rendering: blue isosurface envelope ------------------
    if not args.no_isosurface:
        iso_val = args.iso_fraction * float(sm_iso.max())
        iso_img = create_vtk_image_from_numpy(sm_iso, spacing, origin)
        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(iso_img)
        contour.ComputeNormalsOn()
        contour.SetValue(0, iso_val)
        cmap = vtk.vtkPolyDataMapper()
        cmap.SetInputConnection(contour.GetOutputPort())
        cmap.ScalarVisibilityOff()
        iso = vtk.vtkActor()
        iso.SetMapper(cmap)
        p = iso.GetProperty()
        p.SetColor(*args.iso_color)
        p.SetOpacity(args.iso_opacity)
        p.SetAmbient(0.6)
        p.SetDiffuse(0.6)
        p.SetSpecular(0.5)
        p.SetSpecularPower(40)
        actors.append(iso)

    # 3. Green plume centreline ----------------------------------------------
    if not args.no_centerline:
        pts, _ = extract_centerline(sm_iso, spacing, origin,
                                    threshold=args.centerline_threshold * float(sm_iso.max()),
                                    smooth_window=args.centerline_window)
        if len(pts) >= 2:
            vp = vtk.vtkPoints()
            for pt in pts:
                vp.InsertNextPoint(*pt)
            poly = vtk.vtkPolyData()
            poly.SetPoints(vp)
            lines = vtk.vtkCellArray()
            lines.InsertNextCell(len(pts))
            for i in range(len(pts)):
                lines.InsertCellPoint(i)
            poly.SetLines(lines)
            tube = vtk.vtkTubeFilter()
            tube.SetInputData(poly)
            tube.SetRadius(args.centerline_radius)
            tube.SetNumberOfSides(10)
            tube.CappingOn()
            tm = vtk.vtkPolyDataMapper()
            tm.SetInputConnection(tube.GetOutputPort())
            centre = vtk.vtkActor()
            centre.SetMapper(tm)
            centre.GetProperty().SetColor(*args.centerline_color)
            centre.GetProperty().SetAmbient(0.6)
            centre.GetProperty().SetDiffuse(0.6)
            actors.append(centre)

    bounds = (origin[0], origin[0] + (shape[0] - 1) * spacing[0],
              origin[1], origin[1] + (shape[1] - 1) * spacing[1],
              origin[2], origin[2] + (shape[2] - 1) * spacing[2])
    return actors, bounds


def buoyant_velocity(sm_iso, spacing, origin, rise=1.0, entrain=0.25):
    """Synthetic buoyant-plume velocity field (a stand-in for the paper's Doppler
    field, which the contest does not release).

    Fluid rises where backscatter is strong and is entrained toward the local
    plume centre, giving the vertical, slightly converging streamlines that
    Figure 5 of the paper shows.
    """
    nx, ny, nz = sm_iso.shape
    xs = origin[0] + np.arange(nx) * spacing[0]
    ys = origin[1] + np.arange(ny) * spacing[1]
    X, Y = np.meshgrid(xs, ys, indexing="ij")

    xc = np.empty(nz)
    yc = np.empty(nz)
    for k in range(nz):
        sl = sm_iso[:, :, k]
        ix, iy = np.unravel_index(int(np.argmax(sl)), sl.shape)
        xc[k] = origin[0] + ix * spacing[0]
        yc[k] = origin[1] + iy * spacing[1]

    peak = float(sm_iso.max()) or 1.0
    strength = gaussian_filter(np.clip(sm_iso / peak, 0.0, 1.0), 1.0)
    # Confine the field to the plume so streamlines terminate at its boundary
    # instead of running off into the empty domain.
    strength = np.where(strength >= 0.025, strength, 0.0)

    vx = (-entrain * (X[:, :, None] - xc[None, None, :]) * strength).astype(np.float32)
    vy = (-entrain * (Y[:, :, None] - yc[None, None, :]) * strength).astype(np.float32)
    vz = (rise * strength).astype(np.float32)
    return vx, vy, vz, strength


def plume_base_center(sm_iso, spacing, origin, z=1.0):
    k = int(np.clip(round((z - origin[2]) / spacing[2]), 0, sm_iso.shape[2] - 1))
    sl = sm_iso[:, :, k]
    ix, iy = np.unravel_index(int(np.argmax(sl)), sl.shape)
    return (origin[0] + ix * spacing[0], origin[1] + iy * spacing[1],
            origin[2] + k * spacing[2])


def add_streamlines(actors, sm_iso, spacing, origin, args):
    """Figure-5 style RK4 streamlines of the velocity field, coloured by speed."""
    from pyviz4d import vector_field_to_vtk, trace_streamlines

    vx, vy, vz, strength = buoyant_velocity(sm_iso, spacing, origin,
                                            rise=args.rise, entrain=args.entrain)
    field = vector_field_to_vtk(vx, vy, vz, spacing=spacing, origin=origin)

    # Seed uniformly across the plume body so streamlines fill its width.
    mask = np.argwhere(strength >= 0.06)
    rng = np.random.default_rng(0)
    pick = rng.choice(len(mask), size=min(args.n_seeds, len(mask)), replace=False)
    seeds = vtk.vtkPoints()
    for i, j, k in mask[pick]:
        seeds.InsertNextPoint(origin[0] + i * spacing[0],
                              origin[1] + j * spacing[1],
                              origin[2] + k * spacing[2])
    sp = vtk.vtkPolyData()
    sp.SetPoints(seeds)

    lines = trace_streamlines(field, sp, direction="forward",
                              max_propagation=args.max_propagation,
                              initial_step=spacing[0] * 0.5,
                              terminal_speed=1e-6)

    calc = vtk.vtkArrayCalculator()
    calc.SetInputData(lines)
    calc.AddVectorArrayName("vectors")
    calc.SetFunction("mag(vectors)")
    calc.SetResultArrayName("speed")
    calc.Update()
    speed = calc.GetOutput().GetPointData().GetArray("speed")
    span = speed.GetRange() if speed is not None else (0.0, 1.0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(calc.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("speed")
    mapper.SetScalarRange(*span)
    mapper.SetLookupTable(matplotlib_ctf(args.streamline_colormap, span[0], span[1], n=128))
    mapper.ScalarVisibilityOn()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetLineWidth(args.streamline_width)
    actor.GetProperty().SetAmbient(0.6)
    actor.GetProperty().SetDiffuse(0.6)
    actors.append(actor)
    return span


def position_camera(ren, bounds, zoom=1.30):
    """Near-side 3/4 view so the rising plume is upright and fills the frame."""
    x0, x1, y0, y1, z0, z1 = bounds
    cam = ren.GetActiveCamera()
    cam.SetPosition(x1 + 0.60 * (x1 - x0), y0 - 1.05 * (y1 - y0), z0 + 0.75 * (z1 - z0))
    cam.SetFocalPoint(0.5 * (x0 + x1) + 0.6, 0.5 * (y0 + y1), 0.45 * (z0 + z1))
    cam.SetViewUp(0, 0, 1)
    ren.ResetCamera()
    ren.GetActiveCamera().Zoom(zoom)
    ren.ResetCameraClippingRange()


def render(actors, bounds, args, path, annotate=True):
    ren = vtk.vtkRenderer()
    ren.SetBackground(*args.background)
    ren.SetUseDepthPeeling(1)
    ren.SetOcclusionRatio(0.0)
    ren.SetMaximumNumberOfPeels(16)
    ren.SetUseHiddenLineRemoval(0)
    for a in actors:
        ren.AddActor(a)

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.SetSize(*args.size)
    rw.SetAlphaBitPlanes(1)
    rw.SetMultiSamples(8)
    rw.AddRenderer(ren)

    position_camera(ren, bounds, zoom=args.zoom)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(path)
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    print(f"Wrote {path}")

    if not args.no_axis and annotate:
        annotate_height_scale(ren, bounds, args, path)


def annotate_height_scale(ren, bounds, args, path):
    """Draw the paper's left-hand 'Plume Height (m)' scale in 2D.

    Tick positions come from the renderer's world->display projection, so the
    scale matches the projected height. (The paper likewise post-processes its
    figures -- its Figure 6 colormaps are exported to Matplotlib.)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not available; skipping the height scale.")
        return

    x0, x1, y0, y1, z0, z1 = bounds
    xc, yc = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    ren.SetWorldPoint(xc, yc, z0, 1.0)
    ren.WorldToDisplay()
    y_base = args.size[1] - ren.GetDisplayPoint()[1]
    ren.SetWorldPoint(xc, yc, z1, 1.0)
    ren.WorldToDisplay()
    y_top = args.size[1] - ren.GetDisplayPoint()[1]

    img = Image.open(path).convert("RGB")
    dr = ImageDraw.Draw(img)
    axis_x = int(0.10 * args.size[0])
    ink = (25, 25, 25)
    dr.line([(axis_x, y_base), (axis_x, y_top)], fill=ink, width=3)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24)
    except Exception:
        font = ImageFont.load_default()

    tick = np.floor(z0)
    z = tick
    while z <= z1 + 1e-6:
        yy = y_base + (y_top - y_base) * ((z - z0) / (z1 - z0))
        dr.line([(axis_x, yy), (axis_x + 14, yy)], fill=ink, width=3)
        dr.text((axis_x + 22, yy - 13), f"{z:.0f}", fill=(15, 15, 15), font=font)
        z += args.tick_step

    dr.text((0.05 * args.size[0], 0.02 * args.size[1]), "Plume Height (m)",
            fill=(10, 10, 10), font=font)
    img.save(path)
    print(f"Annotated {path}")


# --------------------------------------------------------------------------- #

def stitch_side_by_side(panels, out, gap=18):
    """Combine rendered panels into one image with a caption under each
    (Figure 6 of the paper lays the two colormaps out side by side)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not available; cannot stitch the colormap comparison.")
        return
    ims = [Image.open(p).convert("RGB") for p, _ in panels]
    W = sum(i.width for i in ims) + gap * (len(ims) - 1)
    H = max(i.height for i in ims) + 46
    canvas = Image.new("RGB", (W, H), (205, 205, 205))
    dr = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    x = 0
    for im, (_, name) in zip(ims, panels):
        canvas.paste(im, (x, 0))
        dr.text((x + 14, im.height + 8), name, fill=(20, 20, 20), font=font)
        x += im.width + gap
    canvas.save(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data/covis")
    p.add_argument("--frame", type=int, default=2, help="index into the sorted .mat files")
    p.add_argument("--roi", nargs=6, type=float, default=[-3.5, 7.5, -18.0, -10.0, 0.0, 10.5],
                   metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))

    p.add_argument("--colormap", default="plume_gold")
    p.add_argument("--segment-threshold", type=float, default=1e-5)
    p.add_argument("--sigma-iso", type=float, default=2.2, help="smoothing for envelope + centreline")
    p.add_argument("--sigma-display", type=float, default=1.0, help="smoothing for the volume")
    p.add_argument("--prune-frac", type=float, default=0.30, help="drop components below this frac of the largest")
    p.add_argument("--opacity-power", type=float, default=1.0)
    p.add_argument("--max-opacity", type=float, default=0.92)

    p.add_argument("--iso-fraction", type=float, default=0.010,
                   help="envelope level as a fraction of the smoothed field max")
    p.add_argument("--iso-color", nargs=3, type=float, default=[0.55, 0.70, 0.97])
    p.add_argument("--iso-opacity", type=float, default=0.17)
    p.add_argument("--no-isosurface", action="store_true")

    p.add_argument("--centerline-threshold", type=float, default=0.002,
                   help="threshold as a fraction of the smoothed field max")
    p.add_argument("--centerline-window", type=int, default=21)
    p.add_argument("--centerline-radius", type=float, default=0.08)
    p.add_argument("--centerline-color", nargs=3, type=float, default=[0.10, 0.80, 0.20])
    p.add_argument("--no-centerline", action="store_true")

    p.add_argument("--background", nargs=3, type=float, default=[0.52, 0.52, 0.52])
    p.add_argument("--size", nargs=2, type=int, default=[960, 1200])
    p.add_argument("--zoom", type=float, default=1.30)
    p.add_argument("--no-axis", action="store_true")
    p.add_argument("--tick-step", type=float, default=2.0)

    p.add_argument("--mode", choices=["volume", "streamlines", "colormaps"], default="volume",
                   help="'volume' = Figure 3; 'streamlines' = Figure-5-style velocity field; "
                        "'colormaps' = Figure-6-style grey vs rainbow comparison")
    p.add_argument("--compare-colormap", default="gist_rainbow",
                   help="second colormap for --mode colormaps")
    p.add_argument("--rise", type=float, default=1.0)
    p.add_argument("--entrain", type=float, default=0.25)
    p.add_argument("--n-seeds", type=int, default=160)
    p.add_argument("--seed-radius", type=float, default=3.0)
    p.add_argument("--seed-height", type=float, default=1.0)
    p.add_argument("--max-propagation", type=float, default=30.0)
    p.add_argument("--streamline-colormap", default="jet")
    p.add_argument("--streamline-width", type=float, default=1.5)

    p.add_argument("--out", default="examples/plumeviz_fig3.png")
    args = p.parse_args()

    register_plume_colormaps()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.mat")))
    if not files:
        raise SystemExit(f"No COVIS .mat files in {args.data_dir!r}. "
                         "Download them from https://sciviscontest2024.github.io/data/")
    path = files[min(max(args.frame, 0), len(files) - 1)]

    vol, spacing, origin, name = load_covis(path, args.roi)
    print(f"Loaded {name}  ROI shape={vol.shape}")

    log_display, sm_iso, sm_disp = prepare(vol, args.segment_threshold,
                                           args.sigma_iso, args.sigma_display,
                                           args.prune_frac)
    actors, bounds = build_scene(log_display, sm_iso, spacing, origin, vol.shape, args)
    if args.mode == "streamlines":
        span = add_streamlines(actors, sm_iso, spacing, origin, args)
        print(f"Streamline speed range [0, {span[1]:.3g}]")
        render(actors, bounds, args, args.out)
        return

    if args.mode == "colormaps":
        panels = []
        for cmap in (args.colormap, args.compare_colormap):
            args.colormap = cmap
            a, b = build_scene(log_display, sm_iso, spacing, origin, vol.shape, args)
            tmp = os.path.join(os.path.dirname(args.out) or ".",
                               f"._cmp_{cmap}.png")
            render(a, b, args, tmp, annotate=False)
            panels.append((tmp, cmap))
        stitch_side_by_side(panels, args.out)
        for tmp, _ in panels:
            os.remove(tmp)
        print(f"Wrote {args.out}")
        return

    render(actors, bounds, args, args.out)


if __name__ == "__main__":
    main()
