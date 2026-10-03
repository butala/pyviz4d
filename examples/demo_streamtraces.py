"""Colourful streamtraces through a baffled contact tank.

    uv run --extra dev python examples/demo_streamtraces.py [--interactive]

The reference here is the CFD streamtrace figure used in hydraulic
engineering -- Angeloudis et al., *Flow, transport and disinfection performance
in small- and full-scale contact tanks*, Figure 5: "streamtraces coloured with
regards to their velocity magnitude".  A contact tank is a serpentine baffled
channel: water enters, threads between the baffles, and leaves.  What makes the
picture is that the flow *structure* is legible -- one fast jet, slow
recirculation eddies in the dead zones behind each baffle -- and the colour
carries the speed.

Three things separate that from "spaghetti", and all three are about the
field rather than the renderer:

1. **The field is steady and smooth.**  Random seeding in a turbulent field
   gives lines that have to tangle.  Here the velocity is an analytic sum of a
   meandering jet, a weak secondary roll and a few point vortices, so a traced
   line is a smooth curve and two lines can never cross.
2. **Seeds come from a rake**, one curtain at the inlet, so what you see is a
   *cross-section of the flow* -- the ribbons fan, fold and return in an
   ordered way.
3. **The lines are long.**  They run the length of the domain, so each one
   tells a story from inlet to outlet instead of stopping mid-thought.

Coloured by |v| through turbo, shaded tubes, on black.
"""
import argparse

import numpy as np
import vtk

from pyviz4d import Viewer4D, render_to_png, viewpoint
from pyviz4d.streamline import StreamlineActor

# Tank: a shallow channel with three baffles forcing a serpentine path.
L, W, H = 12.0, 6.0, 2.0                 # metres
BAFFLES = ((3.0, +1), (6.0, -1), (9.0, +1))   # (x, +1 from the south wall)


def flow_field(nx=120, ny=60, nz=28):
    """Velocity on a regular grid: meandering jet + secondary roll + eddies.

    Returned as three (nx, ny, nz) arrays.  Analytic, so it is smooth and
    steady and a streamline through it is a smooth curve.
    """
    x = np.linspace(0.0, L, nx)[:, None, None]
    y = np.linspace(0.0, W, ny)[None, :, None]
    z = np.linspace(0.0, H, nz)[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)
    x, y, z = x.copy(), y.copy(), z.copy()

    # -- the serpentine jet: its centreline meanders and the flow follows it
    k = 2.0 * np.pi / 6.0
    yc = 0.5 * W + 1.3 * np.sin(k * x)
    dyc_dx = 1.3 * k * np.cos(k * x)
    core = np.exp(-((y - yc) / 0.85) ** 2)
    u = 0.22 + 1.8 * core                      # slow bulk, fast core
    v = u * dyc_dx                             # turn with the meander

    # -- recirculation: a counter-rotating pair in each baffle wake
    for bx, side in BAFFLES:
        for s in (+1.0, -1.0):
            cx = bx + 0.55 * s
            cy = 0.5 * W - side * 1.9
            g = s * side * 0.95
            r2 = (x - cx) ** 2 + (y - cy) ** 2 + 0.08
            u += -g * (y - cy) / r2
            v += g * (x - cx) / r2

    # -- shallow layer: no-slip at bed and surface, plus a weak helical roll
    #    through the bends so the ribbons are genuinely three-dimensional
    profile = 4.0 * z * (H - z) / (H * H)      # parabolic, peaks mid-depth
    u = u * profile
    v = v * profile
    w = 0.28 * core * profile * np.sin(k * x) * np.cos(np.pi * z / H)
    return (u * 0.55).astype(np.float32), (v * 0.55).astype(np.float32), \
        (w * 0.55).astype(np.float32)


def inlet_rake(ny=24, nz=8, jitter=0.06, seed=3):
    """A curtain of seeds filling the inlet cross-section.

    Spread across the *whole* section, with a touch of jitter.  A narrow rake
    sends every ribbon down the same path and the picture becomes a rope; the
    point of a streamtrace figure is that the ribbons separate and fold, which
    is what reveals the jet and the eddies behind the baffles.
    """
    rng = np.random.default_rng(seed)
    pts = vtk.vtkPoints()
    for yy in np.linspace(0.35, W - 0.35, ny):
        for zz in np.linspace(0.18 * H, 0.82 * H, nz):
            pts.InsertNextPoint(0.7 + rng.normal(0, jitter),
                               yy + rng.normal(0, jitter),
                               zz + rng.normal(0, jitter * 0.5))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    return pd


def tank_edges():
    """The tank and baffle outlines, as thin grey tubes for context."""
    segs = []
    for (x0, y0, z0), (x1, y1, z1) in (
            ((0, 0, 0), (L, 0, 0)), ((L, 0, 0), (L, W, 0)),
            ((L, W, 0), (0, W, 0)), ((0, W, 0), (0, 0, 0)),
            ((0, 0, H), (L, 0, H)), ((L, 0, H), (L, W, H)),
            ((L, W, H), (0, W, H)), ((0, W, H), (0, 0, H)),
            ((0, 0, 0), (0, 0, H)), ((L, 0, 0), (L, 0, H)),
            ((L, W, 0), (L, W, H)), ((0, W, 0), (0, W, H))):
        segs.append(((x0, y0, z0), (x1, y1, z1)))
    for bx, side in BAFFLES:
        y0 = 0.0 if side > 0 else W - 2.1
        y1 = 2.1 if side > 0 else W
        segs += [((bx, y0, 0), (bx, y0, H)), ((bx, y1, 0), (bx, y1, H)),
                 ((bx, y0, H), (bx, y1, H)), ((bx, y0, 0), (bx, y1, 0))]
    pts = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    for a, b in segs:
        i = pts.InsertNextPoint(*a)
        j = pts.InsertNextPoint(*b)
        idl = vtk.vtkIdList()
        idl.InsertNextId(i)
        idl.InsertNextId(j)
        lines.InsertNextCell(idl)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetLines(lines)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.35, 0.38, 0.45)
    actor.GetProperty().SetLineWidth(1.5)
    return actor


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-seeds-y", type=int, default=24)
    p.add_argument("--n-seeds-z", type=int, default=8)
    p.add_argument("--tube-radius", type=float, default=0.032)
    p.add_argument("--length", type=float, default=46.0,
                   help="how far each ribbon is traced")
    p.add_argument("--png", default="docs/streamtraces.png")
    p.add_argument("--size", nargs=2, type=int, default=[1400, 900])
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    u, v, w = flow_field()
    vec = [(u, v, w)]   # a single steady frame: StreamlineActor takes a list
    spacing = (L / (u.shape[0] - 1), W / (u.shape[1] - 1), H / (u.shape[2] - 1))
    speed = np.sqrt(u * u + v * v + w * w)
    print(f"field {u.shape}  |v| {speed.min():.3f}..{speed.max():.3f} "
          f"p50 {np.median(speed):.3f}  (m/s)")

    actor = StreamlineActor(vec, spacing=spacing,
                            seeds=inlet_rake(args.n_seeds_y, args.n_seeds_z),
                            direction="both", max_propagation=args.length,
                            initial_step=0.15, max_steps=4000,
                            color_by_magnitude=True, colormap="turbo",
                            magnitude_range=(float(np.percentile(speed, 5)),
                                             float(np.percentile(speed, 98))),
                            tube_radius=args.tube_radius, line_width=1.0)
    prop = actor.actor.GetProperty()
    prop.SetAmbient(0.45)                  # tubes need roundness to read as 3-D
    prop.SetDiffuse(0.75)
    prop.SetSpecular(0.45)
    prop.SetSpecularPower(30)

    actors = [actor.actor, tank_edges()]

    cam = vtk.vtkCamera()
    cam.SetFocalPoint(0.5 * L, 0.5 * W, 0.45 * H)
    cam.SetPosition(0.5 * L + 1.1 * L, -0.85 * L, 0.55 * L)
    cam.SetViewUp(0, 0, 1)

    if args.interactive:
        viewer = Viewer4D(size=(args.size[0], args.size[1]),
                          bg_color=(0.02, 0.02, 0.04))
        viewer.ren_win.SetMultiSamples(8)
        for a in actors:
            viewer.add_actor(a)
        viewpoint(viewer.ren, cam)
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
        viewer.start()
        return 0

    render_to_png(actors, args.png, size=(args.size[0], args.size[1]),
                  camera=cam, bg_color=(0.02, 0.02, 0.04), zoom=1.35)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
