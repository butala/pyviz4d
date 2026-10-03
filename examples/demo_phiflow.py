"""A visually striking 4-D smoke plume: PhiFlow in, pyviz4D out.

    uv run --extra phiflow python examples/demo_phiflow.py            # still PNG
    uv run --extra phiflow python examples/demo_phiflow.py --interactive

Three spatial dimensions plus time: a buoyant plume is integrated forward in a
box by [PhiFlow](https://github.com/tum-pbs/PhiFlow) (BSD-3, Holl et al. 2019)
-- MacCormack advection for the density and the velocity, a Boussinesq buoyancy
term, and a sparse CG projection for incompressibility.  That is a real
simulation, not a canned dataset: re-run it with a different ``--frames`` or
``--res`` and you get a different plume.

pyviz4d then renders every time step three ways at once, which is what makes
the result read as volume rather than as geometry:

* a **ray-cast volume** on a blackbody ramp (near-black shadow -> ember ->
  orange -> gold -> white hot), with **gradient opacity** so the flat interior
  stays see-through and only the edges of the wisps accumulate -- the single
  biggest lever between "fog" and "smoke";
* **translucent isosurface shells** in electric cyan, a cool foil to the fire;
* **streamlines** through the velocity field, coloured by speed, so the
  vortices that give the plume its curl are visible rather than implied.

``--interactive`` opens a Viewer4D window and animates the run with the time
slider (space to pause); without it you get an offscreen still of a chosen
``--frame``.

Ported from the earlier isosurface-only sketch, which ray-cast nothing and
therefore looked like a stack of coloured bubbles.
"""
import argparse

import numpy as np
import vtk

from pyviz4d import (
    IsosurfaceActor,
    StreamlineActor,
    Viewer4D,
    VolumeActor,
    gradient_field,
    render_to_png,
    viewpoint,
)


def blackbody(n=256):
    """A sunset ramp: indigo shadow -> violet -> magenta -> coral -> cream.

    Returned as ``[(scalar, r, g, b), ...]`` for VolumeActor's ``color_points``.

    Deliberately *not* blackbody.  A fire ramp puts dark orange in the middle
    of its range, and semi-transparent dark orange composited over a near-black
    ground is brown -- which is what most of the volume ends up being.  Keeping
    the whole low half cool (indigo/violet) means the thin smoke reads blue and
    only the dense core turns warm, where bright coral and cream cannot go
    muddy.  The hue path also never enters 30-70 deg at low value, which is
    where olive and brown live.
    """
    stops = [(0.00, (0.03, 0.02, 0.12)),   # indigo shadow
             (0.22, (0.16, 0.08, 0.52)),   # violet
             (0.45, (0.55, 0.14, 0.80)),   # purple-magenta
             (0.64, (0.95, 0.28, 0.62)),   # magenta
             (0.82, (1.00, 0.58, 0.46)),   # coral
             (1.00, (1.00, 0.96, 0.88))]   # cream
    xs = np.linspace(0.0, 1.0, n)
    out = []
    for t in xs:
        for (a0, c0), (a1, c1) in zip(stops, stops[1:]):
            if a0 <= t <= a1:
                u = (t - a0) / (a1 - a0) if a1 > a0 else 0.0
                r, g, b = (c0[k] + u * (c1[k] - c0[k]) for k in range(3))
                out.append((t, r, g, b))
                break
    return out


def simulate(res=28, frames=140, warmup=60, buoyancy=0.35, dissipation=0.055,
             drag=0.03, verbose=True):
    """Integrate a buoyant plume; return (density frames, velocity frames).

    ``warmup`` steps are integrated and thrown away, and ``dissipation`` bleeds
    density each step.  Both matter for the same reason: without dissipation
    the closed box accumulates mass without bound (the source never stops) and
    the plume turns into a rectangular solid of smoke by the last frames, and
    without a warm-up the early frames are an almost empty box.  Together they
    put the *recorded* sequence straight into the quasi-steady regime, so the
    whole animation is legible and one transfer function suits every frame.

    ``density[i]`` is an (nx, ny, nz) array; ``velocity[i]`` is a (3, nx, ny,
    nz) array of the collocated velocity, sampled at cell centres so it can be
    fed straight to a stream tracer.
    """
    from phi.flow import (
        Box,
        CenteredGrid,
        Solve,
        Sphere,
        StaggeredGrid,
        advect,
        extrapolation,
        fluid,
        vec,
    )

    bounds = Box(x=(0, 100), y=(0, 200), z=(0, 100))
    shape = dict(x=res, y=res * 2, z=res)
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY, bounds=bounds, **shape)
    density = CenteredGrid(0, extrapolation.BOUNDARY, bounds=bounds, **shape)
    # A slightly off-centre source: perfectly symmetric plumes look synthetic.
    inflow = 0.35 * CenteredGrid(Sphere(x=46, y=8, z=54, radius=9),
                                 extrapolation.BOUNDARY, bounds=bounds, **shape)

    dens, vels = [], []
    total = warmup + frames
    done = 0
    try:
        from tqdm import trange
        steps = trange(total, desc="simulating")
    except ImportError:                        # tqdm is an extra, not a dep
        steps = range(total)

    for _ in steps:
        density = advect.mac_cormack(density, velocity, dt=1.0) + inflow
        density = density * (1.0 - dissipation)   # bleed mass out; the box is
        # closed (BOUNDARY extrapolation) and the source never stops, so with
        # this the field would grow until the whole box is opaque
        velocity = advect.mac_cormack(velocity, velocity, dt=1.0) \
            + (density * vec(x=0, y=buoyancy, z=0)).at(velocity)
        velocity = velocity * (1.0 - drag)   # damp box-scale sloshing.  A
        # closed cavity heated from below oscillates (the whole cell rolls
        # over), and that breathing is what makes the plume swell to a slab
        # and then vanish between frames.
        velocity, _ = fluid.make_incompressible(
            velocity, solve=Solve('CG', rel_tol=1e-3, abs_tol=1e-3,
                                  max_iterations=2000))

        done += 1
        if done > warmup:
            dens.append(density.values.numpy('x,y,z'))
            v = velocity.at(density).values.numpy('vector,x,y,z')
            vels.append(np.asarray(v, dtype=np.float32))

    if verbose:
        d = np.asarray(dens)
        print(f"simulated {frames} frames at {res}x{2*res}x{res} after "
              f"{warmup} warm-up steps ({d.size/1e6:.1f}M voxels); "
              f"density p50 {np.percentile(d, 50):.3f} p99 {np.percentile(d, 99):.3f} "
              f"max {d.max():.3f}")
    return dens, vels


def normalise_display(dens, floor_frac=0.35):
    """Scale every frame into one common display range.

    A single transfer function has to serve 140 frames whose peak density
    varies by more than an order of magnitude.  Scale it off a global peak and
    the burst frames saturate into a solid slab while the quiet ones vanish --
    which is exactly how the first version read ("wispy at the beginning, a
    rectangular solid at the end").  Each frame is brought to the *median*
    frame's peak instead, with a floor so a nearly empty frame is amplified by
    at most 1/floor_frac and cannot turn into grain.  The plume's own time
    evolution survives -- only the display range is levelled.
    """
    peaks = [float(np.percentile(d, 99.0)) for d in dens]
    ref = float(np.median(peaks)) or 1.0
    return [d * (ref / max(pk, floor_frac * ref)) for d, pk in zip(dens, peaks)]


def build_scene(dens, vels, spacing, args):
    """Volume + shells + streamlines, as TemporalActors so they animate."""
    dens = normalise_display(dens)
    actors = []
    hi = float(np.percentile(dens, 99.5))
    # Deliberately gentle: at 0.97 the plume occluded itself into a solid.  The
    # cores may read pale, but you can see *into* the volume at every depth.

    # 1. Ray-cast volume, blackbody, gradient opacity -------------------------
    vol = VolumeActor(dens, spacing=spacing, color_points=blackbody(),
                      opacity_points=[(0.0, 0.0), (0.15 * hi, 0.05),
                                      (0.50 * hi, 0.18), (hi, 0.45)])
    gx, gy, gz = gradient_field(dens[0], spacing)
    g_hi = float(np.percentile(np.sqrt(gx * gx + gy * gy + gz * gz), 99.0)) or 1.0
    gop = vtk.vtkPiecewiseFunction()
    gop.AddPoint(0.0, 0.0)
    gop.AddPoint(0.30 * g_hi, 0.18)
    gop.AddPoint(g_hi, 0.85)
    vol.prop.SetGradientOpacity(gop)             # flat interior -> see-through
    vol.prop.SetSpecular(0.5)
    vol.prop.SetSpecularPower(48)
    vol.mapper.SetBlendModeToComposite()
    vol.mapper.SetAutoAdjustSampleDistances(1)
    vol.mapper.SetSampleDistance(float(min(spacing)) * 0.4)
    actors.append(vol)

    # 2. Translucent shells: a cool foil to the fire -------------------------
    shells = IsosurfaceActor(dens, spacing=spacing,
                             iso_values=[0.6 * hi, 1.4 * hi],
                             colors=[(0.35, 0.85, 0.95), (0.75, 0.95, 1.0)],
                             opacity=0.08)
    actors.append(shells)

    # 3. Vortex-core ribbons, coloured by speed ------------------------------
    # Seeded on |curl v| maxima and kept short, so what you see is the ring
    # under the cap and the braids along the stem rather than a tangle.  Ice
    # ramp + shaded tubes: the cool complement to the volume's violet/coral,
    # so the flow reads as a distinct layer instead of more of the same.
    register_accent()
    vec_frames = [tuple(v[i] for i in range(3)) for v in vels]
    speeds = [float(np.linalg.norm(v, axis=0).max()) for v in vels]
    ribbons = StreamlineActor(
        vec_frames, spacing=spacing,
        seeds=vortex_seeds(vels[len(vels) // 2], spacing, n=args.n_seeds),
        direction="forward", max_propagation=args.max_propagation,
        initial_step=1.0, color_by_magnitude=True, colormap="smoke_lime",
        magnitude_range=(0.0, float(np.percentile(speeds, 90)) or 1.0),
        tube_radius=args.tube_radius, line_width=1.0)
    # Self-lit, like an overlay: a shaded tube is a diffuse surface and sinks
    # straight into the volume in front of it.  This is an annotation of the
    # flow, so it should read as one.
    rp = ribbons.actor.GetProperty()
    rp.SetAmbient(1.0)
    rp.SetDiffuse(0.15)
    rp.SetSpecular(0.25)
    actors.append(ribbons)
    return actors, hi


def register_accent():
    """A lime accent ramp -- deep green -> green -> spring -> pale lime.

    Registered so volume.matplotlib_ctf can sample it by name.  The volume is
    a sunset (indigo -> violet -> magenta -> coral -> cream) and its thin end
    is violet, i.e. *blue-dominant* -- so both `plasma` and a cyan accent sit
    on the same hue as the smoke and disappear into it.  Green is the one hue
    the scene does not use, so the flow reads as a distinct layer.  (This is
    the same reason the paper's figures put a green centreline in a gold
    plume.)
    """
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    if "smoke_lime" not in matplotlib.colormaps():
        matplotlib.colormaps.register(LinearSegmentedColormap.from_list("smoke_lime", [
            # The floor is a *visible* green, not near-black: a ramp that
            # starts dark makes every slow segment of every ribbon vanish, and
            # in a plume most of the length is slow -- which is the other half
            # of "the streamlines are not adding a whole lot".
            (0.00, (0.10, 0.38, 0.12)),
            (0.35, (0.28, 0.75, 0.16)),
            (0.70, (0.62, 0.98, 0.30)),
            (1.00, (0.95, 1.00, 0.85)),
        ]))


def vorticity(v):
    """|curl v| for a (3, nx, ny, nz) velocity array on a regular grid."""
    vx, vy, vz = (np.asarray(v[i], dtype=np.float32) for i in range(3))
    wx = np.gradient(vz, axis=1) - np.gradient(vy, axis=2)
    wy = np.gradient(vx, axis=2) - np.gradient(vz, axis=0)
    wz = np.gradient(vy, axis=0) - np.gradient(vx, axis=1)
    return np.sqrt(wx * wx + wy * wy + wz * wz)


def vortex_seeds(v, spacing, n=32, min_sep=4):
    """Seeds placed on the vortex cores, not scattered through the smoke.

    Random seeding through the density mass puts a line in every wisp, which
    is what spaghetti is.  The structures worth drawing in a plume are the
    vortex cores -- the ring under the cap and the braids along the stem --
    and those are the maxima of |curl v|.  Take the strongest of those, with a
    minimum separation so 32 seeds do not all sit in one knot.
    """
    w = vorticity(v)
    order = np.argsort(w.ravel())[::-1]
    pts = vtk.vtkPoints()
    taken = np.zeros(w.shape, dtype=bool)
    for idx in order:
        if pts.GetNumberOfPoints() >= n:
            break
        i, j, k = np.unravel_index(idx, w.shape)
        if taken[max(0, i - min_sep):i + min_sep,
                 max(0, j - min_sep):j + min_sep,
                 max(0, k - min_sep):k + min_sep].any():
            continue
        taken[i, j, k] = True
        pts.InsertNextPoint(i * spacing[0], j * spacing[1], k * spacing[2])
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    return pd


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=int, default=28,
                   help="grid resolution in x and z (y is 2x).  28 is a hard "
                        "ceiling here: PhiFlow's projection builds a sparse "
                        "normal matrix with n^2 entries, and n = 2*res^3 "
                        "crosses int32 at 46340 cells")
    p.add_argument("--frames", type=int, default=140,
                   help="time steps to integrate and animate")
    p.add_argument("--buoyancy", type=float, default=0.4)
    p.add_argument("--warmup", type=int, default=60,
                   help="steps integrated and discarded first, so the "
                        "recorded run starts in the steady regime")
    p.add_argument("--drag", type=float, default=0.03,
                   help="velocity damping per step; suppresses the "
                        "box-scale sloshing that makes the plume breathe")
    p.add_argument("--dissipation", type=float, default=0.055,
                   help="density bled per step; without it the closed box "
                        "accumulates mass until the plume is a slab")
    p.add_argument("--n-seeds", type=int, default=40,
                   help="vortex-core ribbons to draw (sparse reads "
                        "as structure; dense reads as spaghetti)")
    p.add_argument("--max-propagation", type=float, default=26.0,
                   help="ribbon length; short traces the vortex, "
                        "long crosses the box and tangles")
    p.add_argument("--tube-radius", type=float, default=0.9,
                   help="0 draws flat lines instead of shaded tubes")
    p.add_argument("--frame", type=int, default=110,
                   help="which time step to draw for the still PNG")
    p.add_argument("--png", default="docs/smoke4d.png")
    p.add_argument("--size", nargs=2, type=int, default=[1280, 900])
    p.add_argument("--interactive", action="store_true",
                   help="animate in a Viewer4D window instead of writing a PNG")
    args = p.parse_args()

    dens, vels = simulate(res=args.res, frames=args.frames, warmup=args.warmup,
                          buoyancy=args.buoyancy, dissipation=args.dissipation,
                          drag=args.drag)
    spacing = (100.0 / args.res, 200.0 / (args.res * 2), 100.0 / args.res)
    actors, _ = build_scene(dens, vels, spacing, args)

    cam = vtk.vtkCamera()
    cam.SetFocalPoint(50.0, 95.0, 45.0)
    cam.SetPosition(235.0, -40.0, 175.0)
    cam.SetViewUp(0, 1, 0)

    if args.interactive:
        viewer = Viewer4D(size=(args.size[0], args.size[1]),
                          bg_color=(0.015, 0.015, 0.03))
        viewer.ren.SetUseDepthPeeling(1)
        viewer.ren.SetOcclusionRatio(0.05)
        viewer.ren.SetMaximumNumberOfPeels(24)
        viewer.ren_win.SetAlphaBitPlanes(1)
        viewer.ren_win.SetMultiSamples(8)
        for a in actors:
            viewer.add_actor(a)
        viewpoint(viewer.ren, cam)
        viewer.add_playback_ui(max_time=args.frames - 1, loop=True)
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, "
              "space play/pause, q quit")
        viewer.start(timer_interval_ms=33)
        return 0

    # Offscreen still: advance the actors to the requested time first.
    t = min(max(args.frame, 0), len(dens) - 1)
    for a in actors:
        a.update(float(t))
    render_to_png([a.actor for a in actors], args.png,
                  size=(args.size[0], args.size[1]), camera=cam,
                  bg_color=(0.015, 0.015, 0.03), zoom=2.4)
    print(f"wrote {args.png} (frame {t}/{args.frames - 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
