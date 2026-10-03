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
    """Blackbody ramp: cool shadow -> ember -> orange -> gold -> white hot.

    Returned as ``[(scalar, r, g, b), ...]`` for VolumeActor's ``color_points``.
    A single hue would read as a cutout; the ramp through two hues is what
    makes a volume look incandescent.
    """
    stops = [(0.00, (0.010, 0.008, 0.030)),
             (0.18, (0.26, 0.03, 0.14)),
             (0.45, (0.85, 0.28, 0.05)),
             (0.72, (1.00, 0.74, 0.14)),
             (1.00, (1.00, 0.98, 0.90))]
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


def simulate(res=28, frames=140, buoyancy=0.5, verbose=True):
    """Integrate a buoyant plume; return (density frames, velocity frames).

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
    try:
        from tqdm import trange
        steps = trange(frames, desc="simulating")
    except ImportError:                        # tqdm is an extra, not a dep
        steps = range(frames)

    for _ in steps:
        density = advect.mac_cormack(density, velocity, dt=1.0) + inflow
        velocity = advect.mac_cormack(velocity, velocity, dt=1.0) \
            + (density * vec(x=0, y=buoyancy, z=0)).at(velocity)
        velocity, _ = fluid.make_incompressible(
            velocity, solve=Solve('CG', rel_tol=1e-3, abs_tol=1e-3,
                                  max_iterations=2000))

        dens.append(density.values.numpy('x,y,z'))
        v = velocity.at(density).values.numpy('vector,x,y,z')
        vels.append(np.asarray(v, dtype=np.float32))

    if verbose:
        d = np.asarray(dens)
        print(f"simulated {frames} frames at {res}x{2*res}x{res} "
              f"({d.size/1e6:.1f}M voxels); density max {d.max():.3f}")
    return dens, vels


def build_scene(dens, vels, spacing, args):
    """Volume + shells + streamlines, as TemporalActors so they animate."""
    actors = []
    hi = float(np.percentile(dens, 99.5))

    # 1. Ray-cast volume, blackbody, gradient opacity -------------------------
    vol = VolumeActor(dens, spacing=spacing, color_points=blackbody(),
                      opacity_points=[(0.0, 0.0), (0.35 * hi, 0.28),
                                      (hi, 0.97)])
    gx, gy, gz = gradient_field(dens[0], spacing)
    g_hi = float(np.percentile(np.sqrt(gx * gx + gy * gy + gz * gz), 99.0)) or 1.0
    gop = vtk.vtkPiecewiseFunction()
    gop.AddPoint(0.0, 0.0)
    gop.AddPoint(0.30 * g_hi, 0.25)
    gop.AddPoint(g_hi, 1.0)
    vol.prop.SetGradientOpacity(gop)             # flat interior -> see-through
    vol.prop.SetSpecular(0.5)
    vol.prop.SetSpecularPower(48)
    vol.mapper.SetBlendModeToComposite()
    vol.mapper.SetAutoAdjustSampleDistances(1)
    vol.mapper.SetSampleDistance(float(min(spacing)) * 0.4)
    actors.append(vol)

    # 2. Translucent shells: a cool foil to the fire -------------------------
    shells = IsosurfaceActor(dens, spacing=spacing,
                             iso_values=[0.4 * hi, 0.8 * hi, 1.3 * hi],
                             colors=[(0.15, 0.85, 0.95), (0.55, 0.95, 1.0),
                                     (0.9, 1.0, 1.0)],
                             opacity=0.22)
    actors.append(shells)

    # 3. Streamlines coloured by speed ---------------------------------------
    # Advect a fixed set of seeds each frame from the *velocity* field, which is
    # what carries the curl; colour the tracers by |v|.
    vec_frames = [tuple(v[i] for i in range(3)) for v in vels]
    seeds = seed_cloud(dens[0], spacing)
    actors.append(StreamlineActor(vec_frames, spacing=spacing, seeds=seeds,
                                  direction="both", max_propagation=60.0,
                                  initial_step=1.5, color_by_magnitude=True,
                                  colormap="plasma", line_width=1.6))
    return actors, hi


def seed_cloud(dens0, spacing, n=140):
    """Seed points scattered through the plume's mass, not on a grid."""
    rng = np.random.default_rng(7)
    w = np.clip(dens0, 0, None).ravel()
    if w.sum() <= 0:
        w = np.ones_like(w)
    idx = rng.choice(w.size, size=n, p=w / w.sum())
    pts = vtk.vtkPoints()
    for i in idx:
        ix, iy, iz = np.unravel_index(i, dens0.shape)
        pts.InsertNextPoint(ix * spacing[0], iy * spacing[1], iz * spacing[2])
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
    p.add_argument("--buoyancy", type=float, default=0.5)
    p.add_argument("--frame", type=int, default=110,
                   help="which time step to draw for the still PNG")
    p.add_argument("--png", default="docs/smoke4d.png")
    p.add_argument("--size", nargs=2, type=int, default=[1280, 900])
    p.add_argument("--interactive", action="store_true",
                   help="animate in a Viewer4D window instead of writing a PNG")
    args = p.parse_args()

    dens, vels = simulate(res=args.res, frames=args.frames,
                          buoyancy=args.buoyancy)
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
