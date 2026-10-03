"""A live 4-D Monte Carlo in a contact tank -- particles, streamlines, cross-section.

    uv run --extra dev python examples/demo_particle_flow.py

Endless and interactive.  40 000 particles are advected through the flow with
RK4 plus an eddy diffusivity and reborn at the inlet when they leave, so the
cloud is in statistical steady state and simply runs at the frame rate of the
window.  Nothing is pre-rendered and nothing is cached.

What is on screen, and what each thing means:

* **Particles** -- small dots, one per parcel of fluid, coloured by **residence
    time**, i.e. how long since that parcel entered.  The colour bar is there
    because this is a physical quantity with units (seconds), not decoration.
* **Streamlines** -- the shape of the velocity field, thin and neutral so they
    do not compete with the particles' colour.
* **`v` -- velocity cross-section.**  A slice through mid-depth coloured by
    speed |v|, with its own colour bar.  This is both the velocity
    visualisation and the 2-D cross section: it shows where the jet runs and
    where the fluid sits still.
* **`p` / `l` / `c`** -- toggle particles, streamlines, colour bars.

Keys: `v` cross-section, `p` particles, `l` streamlines, `c` colour bars,
`space` pause, `q` quit.  Left-drag rotates while it runs.
"""
import argparse

import numpy as np
import vtk
from demo_contact_tank import H, L, W, tank_edges, velocity_at

from pyviz4d import ParticleCloudActor, StreamlineActor, Viewer4D, render_to_png, viewpoint


def speed_slice(z=0.5 * H, nx=140, ny=70):
    """A mid-depth plane of |v|, as a triangle mesh with per-vertex scalars."""
    xs = np.linspace(0.0, L, nx)
    ys = np.linspace(0.0, W, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    u, v, w = velocity_at(X, Y, np.full_like(X, z))
    sp = np.sqrt(u * u + v * v + w * w)

    pts = vtk.vtkPoints()
    for i in range(nx):
        for j in range(ny):
            pts.InsertNextPoint(X[i, j], Y[i, j], z)
    scal = vtk.vtkFloatArray()
    scal.SetName("speed")
    for j in range(ny):
        for i in range(nx):
            scal.InsertNextTuple1(float(sp[i, j]))

    tris = vtk.vtkCellArray()
    for i in range(nx - 1):
        for j in range(ny - 1):
            a = i * ny + j
            b = (i + 1) * ny + j
            idl = vtk.vtkIdList()
            for k in (a, b, b + 1, a + 1):
                idl.InsertNextId(k)
            tris.InsertNextCell(idl)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(tris)
    pd.GetPointData().SetScalars(scal)   # per-vertex, so POINT data
    rng = (float(sp.min()), float(sp.max()))

    lut = vtk.vtkColorTransferFunction()
    for t, c in ((0.0, (0.05, 0.10, 0.40)), (0.35, (0.05, 0.45, 0.85)),
                 (0.60, (0.10, 0.85, 0.75)), (0.80, (0.95, 0.85, 0.15)),
                 (1.0, (1.00, 0.30, 0.05))):
        lut.AddRGBPoint(rng[0] + t * (rng[1] - rng[0]), *c)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(*rng)
    mapper.SetScalarModeToUsePointData()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(0.75)
    return actor, lut, rng


def age_colourbar(age_max):
    """The particle legend: residence time, in seconds."""
    lut = vtk.vtkColorTransferFunction()
    for t, c in ((0.0, (0.0, 0.0, 0.0)), (0.06, (0.05, 0.10, 0.45)),
                 (0.28, (0.05, 0.45, 0.85)), (0.50, (0.10, 0.85, 0.75)),
                 (0.68, (0.55, 0.95, 0.20)), (0.85, (1.00, 0.80, 0.10)),
                 (1.0, (1.00, 0.25, 0.05))):
        lut.AddRGBPoint(t * age_max, *c)
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle("residence time (s)")
    bar.SetNumberOfLabels(5)
    bar.SetPosition(0.02, 0.12)
    bar.SetPosition2(0.09, 0.62)
    bar.GetTitleTextProperty().SetColor(1, 1, 1)
    bar.GetTitleTextProperty().SetFontSize(14)
    bar.GetLabelTextProperty().SetColor(1, 1, 1)
    bar.GetLabelTextProperty().SetFontSize(12)
    bar.SetAnnotationTextScaling(0)
    bar.GetAnnotationTextProperty().SetColor(1, 1, 1)
    return bar


def speed_colourbar(rng):
    lut = vtk.vtkColorTransferFunction()
    for t, c in ((0.0, (0.05, 0.10, 0.40)), (0.35, (0.05, 0.45, 0.85)),
                 (0.60, (0.10, 0.85, 0.75)), (0.80, (0.95, 0.85, 0.15)),
                 (1.0, (1.00, 0.30, 0.05))):
        lut.AddRGBPoint(rng[0] + t * (rng[1] - rng[0]), *c)
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lut)
    bar.SetTitle("speed |v| (m/s)")
    bar.SetNumberOfLabels(5)
    bar.SetPosition(0.88, 0.12)
    bar.SetPosition2(0.09, 0.62)
    bar.GetTitleTextProperty().SetColor(1, 1, 1)
    bar.GetTitleTextProperty().SetFontSize(14)
    bar.GetLabelTextProperty().SetColor(1, 1, 1)
    bar.GetLabelTextProperty().SetFontSize(12)
    bar.SetAnnotationTextScaling(0)
    bar.GetAnnotationTextProperty().SetColor(1, 1, 1)
    return bar


def streamlines(ny=16, nz=5, seed=3):
    """The velocity field's shape: a curtain of thin neutral ribbons."""
    u, v, w = velocity_at(*np.meshgrid(
        np.linspace(0, L, 110), np.linspace(0, W, 55),
        np.linspace(0, H, 24), indexing="ij"))
    spacing = (L / 109.0, W / 54.0, H / 23.0)

    rng = np.random.default_rng(seed)
    pts = vtk.vtkPoints()
    for yy in np.linspace(0.35, W - 0.35, ny):
        for zz in np.linspace(0.20 * H, 0.80 * H, nz):
            pts.InsertNextPoint(0.7 + rng.normal(0, 0.06),
                               yy + rng.normal(0, 0.06),
                               zz + rng.normal(0, 0.03))
    seeds = vtk.vtkPolyData()
    seeds.SetPoints(pts)

    sl = StreamlineActor([(u, v, w)], spacing=spacing, seeds=seeds,
                         direction="both", max_propagation=46.0,
                         initial_step=0.15, max_steps=4000,
                         color_by_magnitude=False,
                         tube_radius=0.02, line_width=1.0)
    prop = sl.actor.GetProperty()
    prop.SetColor(0.72, 0.80, 0.90)          # neutral: the particles own the colour
    prop.SetOpacity(0.28)
    prop.SetAmbient(0.7)
    prop.SetDiffuse(0.3)
    return sl


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", type=int, default=40000)
    p.add_argument("--dt", type=float, default=0.045)
    p.add_argument("--diffusivity", type=float, default=0.02)
    p.add_argument("--opacity", type=float, default=0.85,
                   help="per-dot alpha; high = crisp dots, low = a haze")
    p.add_argument("--splat-scale", type=float, default=0.40,
                   help="dot size multiplier (0.007 m at 1.0)")
    p.add_argument("--warmup", type=int, default=150)
    p.add_argument("--with-slice", action="store_true",
                   help="in --steps mode, show the velocity cross-section too")
    p.add_argument("--steps", type=int, default=0,
                   help=">0: integrate this many steps and write a still")
    p.add_argument("--png", default="docs/particle_flow.png")
    p.add_argument("--size", nargs=2, type=int, default=[1400, 900])
    args = p.parse_args()

    bounds = ((0.0, L), (0.0, W), (0.0, H))
    inlet = ((0.0, 1.2), (0.3, W - 0.3), (0.15 * H, 0.85 * H))
    cloud = ParticleCloudActor(velocity_at, bounds, inlet,
                               n=args.particles, dt=args.dt,
                               kappa=args.diffusivity, color_by="age",
                               splat_scale=args.splat_scale,
                               opacity=args.opacity, seed=0)
    for _ in range(args.warmup):
        cloud.update(0.0)
    cloud.age_max = float(np.percentile(cloud.age, 95)) or 1.0

    sl = streamlines()
    slice_actor, _, slice_rng = speed_slice()
    age_bar = age_colourbar(cloud.age_max)
    vel_bar = speed_colourbar(slice_rng)
    edges = tank_edges()
    slice_actor.SetVisibility(0)
    vel_bar.SetVisibility(0)

    print(f"{args.particles} particles  age 0..{cloud.age_max:.2f} s  "
          f"|v| {slice_rng[0]:.2f}..{slice_rng[1]:.2f} m/s")
    print("keys: v cross-section   p particles   l streamlines   c colour bars   q quit")

    if args.steps:
        for i in range(args.steps):
            cloud.update(float(i))
        still = [cloud.actor, sl.actor, edges, age_bar]
        if args.with_slice:
            slice_actor.SetVisibility(1)
            vel_bar.SetVisibility(1)
            still = [slice_actor, cloud.actor, sl.actor, edges,
                     age_bar, vel_bar]
        render_to_png(still, args.png, size=(args.size[0], args.size[1]),
                      camera=_cam(), bg_color=(0.01, 0.01, 0.02), zoom=1.3)
        print(f"wrote {args.png}")
        return 0

    viewer = Viewer4D(size=(args.size[0], args.size[1]),
                      bg_color=(0.01, 0.01, 0.02))
    viewer.ren_win.SetMultiSamples(4)
    for a in (cloud, sl, slice_actor, edges):
        viewer.add_actor(a)
    viewer.ren.AddActor2D(age_bar)
    viewer.ren.AddActor2D(vel_bar)
    viewpoint(viewer.ren, _cam())

    state = {"slice": False, "bar": True}

    def on_key(obj, _e):
        k = obj.GetKeySym().lower()
        if k == "v":
            state["slice"] = not state["slice"]
            slice_actor.SetVisibility(state["slice"])
            vel_bar.SetVisibility(state["slice"] and state["bar"])
        elif k == "p":
            cloud.actor.SetVisibility(0 if cloud.actor.GetVisibility() else 1)
        elif k == "l":
            sl.actor.SetVisibility(0 if sl.actor.GetVisibility() else 1)
        elif k == "c":
            state["bar"] = not state["bar"]
            age_bar.SetVisibility(state["bar"])
            vel_bar.SetVisibility(state["bar"] and state["slice"])
        else:
            return
        obj.GetRenderWindow().Render()

    # Viewer4D binds f and r at priority 1.0 and space once playback UI exists;
    # these are all free.
    viewer.iren.AddObserver("KeyPressEvent", on_key)
    print("running for ever -- left-drag rotate, scroll zoom, q quit")
    viewer.start(timer_interval_ms=16)
    return 0


def _cam():
    cam = vtk.vtkCamera()
    cam.SetFocalPoint(0.5 * L, 0.5 * W, 0.45 * H)
    cam.SetPosition(0.5 * L + 1.1 * L, -0.85 * L, 0.55 * L)
    cam.SetViewUp(0, 0, 1)
    return cam


if __name__ == "__main__":
    raise SystemExit(main())
