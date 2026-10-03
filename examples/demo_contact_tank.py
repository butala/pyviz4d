"""Flow, transport and residence time in a baffled contact tank.

    uv run --extra dev python examples/demo_contact_tank.py [--interactive]

A contact tank is a serpentine baffled channel: water enters, threads between
the baffles and leaves, and its job is to keep water in contact with a
disinfectant for long enough.  Everything engineers ask about one is a
*transport* question -- how long does a parcel of fluid stay? -- and the way to
answer it is to release particles and watch them.

So there are three layers here, and they are three views of the same physics:

* **A Monte Carlo particle simulation.**  Particles are released in a slab at
  the inlet and integrated through the velocity field with RK4 plus a Gaussian
  random increment -- an eddy diffusivity, which is what makes this a Monte
  Carlo rather than a streamline plot.  Each records how long it stayed.
* **A volume render of the speed**, which is the layer that "sees the flow":
  it is near zero in the stagnant pools behind the baffles and high where the
  jet squeezes through each gap, so the three constrictions light up and the
  dead zones stay dark.  (Residence time was tried here first and does not
  work as a volume -- see the note in monte_carlo().)
* **Streamlines, deliberately understated** -- thin, translucent and neutral,
  so they give the shape of the field without competing for colour.  They were
  the whole picture before and it was too loud.
* **The particles themselves**, sparse, coloured by how long they had been in
  there -- the residence time lives here and in the printed distribution,
  where it reads.

Reference for the streamtrace language: Angeloudis et al., *Flow, transport
and disinfection performance in small- and full-scale contact tanks*.
"""
import argparse

import numpy as np
import vtk

from pyviz4d import Viewer4D, VolumeActor, point_actor, render_to_png, viewpoint
from pyviz4d.streamline import StreamlineActor

# Tank: a shallow channel with three baffles forcing a serpentine path.
L, W, H = 12.0, 6.0, 2.0
BAFFLES = ((3.0, +1), (6.0, -1), (9.0, +1))     # (x, +1 = from the south wall)


def velocity_at(x, y, z):
    """The flow, in closed form: meandering jet + secondary roll + wake eddies.

    Analytic and steady, so a streamline through it is a smooth curve and two
    streamlines can never cross -- which is the whole reason this reads as
    ribbons rather than spaghetti.  Also cheap enough to evaluate per particle.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    k = 2.0 * np.pi / 6.0
    yc = 0.5 * W + 1.3 * np.sin(k * x)
    dyc_dx = 1.3 * k * np.cos(k * x)
    core = np.exp(-((y - yc) / 0.85) ** 2)
    u = 0.22 + 1.8 * core
    v = u * dyc_dx

    for bx, side in BAFFLES:                      # recirculation in the wakes
        for s in (+1.0, -1.0):
            cx = bx + 0.55 * s
            cy = 0.5 * W - side * 1.9
            g = s * side * 0.95
            r2 = (x - cx) ** 2 + (y - cy) ** 2 + 0.08
            u += -g * (y - cy) / r2
            v += g * (x - cx) / r2

    profile = 4.0 * z * (H - z) / (H * H)         # no-slip at bed and surface
    u = u * profile
    v = v * profile
    w = 0.28 * core * profile * np.sin(k * x) * np.cos(np.pi * z / H)
    return u * 0.55, v * 0.55, w * 0.55


def flow_grid(nx=120, ny=60, nz=28):
    """The same field sampled on a grid, for the stream tracer."""
    xs = np.linspace(0.0, L, nx)[:, None, None]
    ys = np.linspace(0.0, W, ny)[None, :, None]
    zs = np.linspace(0.0, H, nz)[None, None, :]
    x, y, z = np.broadcast_arrays(xs, ys, zs)
    u, v, w = velocity_at(x, y, z)
    spacing = (L / (nx - 1), W / (ny - 1), H / (nz - 1))
    return (np.ascontiguousarray(u, dtype=np.float32),
            np.ascontiguousarray(v, dtype=np.float32),
            np.ascontiguousarray(w, dtype=np.float32)), spacing


def monte_carlo(n=8000, dt=0.02, n_steps=1500, kappa=0.015,
                shape=(90, 45, 22), seed=0):
    """Release particles at the inlet and watch where they go and how long.

    Each particle is integrated with RK4 through the velocity and nudged by a
    Gaussian increment of variance ``2*kappa*dt`` per step -- the eddy
    diffusivity.  Two things come out:

    ``field``    the mean residence time of the fluid in each cell, 0 where
                 nothing visited.  This is the volume that gets rendered.
    ``arrivals`` the time each particle took to leave, i.e. a sample of the
                 residence time distribution -- the number a contact tank is
                 judged by.

    Particles that run out of steps without leaving are still counted in the
    field (they are the fluid stuck in the dead zones) but not in ``arrivals``.
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = shape
    dx, dy, dz = L / nx, W / ny, H / nz

    # a slab of particles at the inlet, spread across the whole section
    p = np.column_stack([np.full(n, 0.7),
                         rng.uniform(0.35, W - 0.35, n),
                         rng.uniform(0.18 * H, 0.82 * H, n)])
    age = np.zeros(n)
    live = np.ones(n, dtype=bool)
    arrivals = []
    age_sum = np.zeros(shape)
    visit = np.zeros(shape)

    def deposit():
        nonlocal visit, age_sum
        i = np.clip((p[:, 0] / dx).astype(int), 0, nx - 1)
        j = np.clip((p[:, 1] / dy).astype(int), 0, ny - 1)
        k = np.clip((p[:, 2] / dz).astype(int), 0, nz - 1)
        flat = (k * ny + j) * nx + i
        visit += np.bincount(flat, minlength=nx * ny * nz).reshape(shape)
        age_sum += np.bincount(flat, weights=age,
                               minlength=nx * ny * nz).reshape(shape)

    for _ in range(n_steps):
        if not live.any():
            break
        q = p[live]
        a = age[live]

        def vel(pt):
            return np.column_stack(velocity_at(pt[:, 0], pt[:, 1], pt[:, 2]))

        k1 = vel(q)
        k2 = vel(q + 0.5 * dt * k1)
        k3 = vel(q + 0.5 * dt * k2)
        k4 = vel(q + dt * k3)
        q = q + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        q += np.sqrt(2.0 * kappa * dt) * rng.normal(size=q.shape)
        a = a + dt

        p[live] = q
        age[live] = a
        deposit()

        out = live.copy()
        idx = np.where(live)[0]
        gone = (p[live, 0] >= L) | (p[live, 0] < 0.0) \
            | (p[live, 1] < 0.0) | (p[live, 1] > W) \
            | (p[live, 2] < 0.0) | (p[live, 2] > H)
        for i, g in zip(idx, gone):
            if g:
                arrivals.append(age[i])
                out[i] = False
        live = out

    # A light box blur.  Depositing at discrete integration steps aliases into
    # visible vertical banding; three passes of a 3-wide box is close enough to
    # a Gaussian to remove it without smearing the jet away.
    # The *volume* is the path density -- how much fluid went through each
    # cell -- and not the mean age.  Mean age is smooth everywhere and has
    # almost no dynamic range where it matters, so as a volume it can only
    # ever be a haze (tried, measured, abandoned).  Path density has exactly
    # the structure that matters: a dense serpentine ribbon where the jet
    # runs, and thin haloes in the dead zones.  That is "seeing the flow".
    # The residence time stays in the particles' colour and in the printed
    # distribution, where it belongs.
    # The volume is |v|, not a residence field.  Four residence-time volumes
    # were tried -- monotonic age, log path density, raw path density with a
    # high floor, and a diverging age -- and every one of them is a *smooth*
    # scalar, so ray-casting it gives an even haze that hides the streamlines
    # instead of showing the flow.  |v| has the contrast a volume needs: it is
    # ~0 in the stagnant pools behind the baffles and high along the jet.  The
    # residence time stays where it reads well -- the particles' colour and
    # the printed distribution below.
    u, v, w = velocity_at(*np.meshgrid(
        (np.arange(shape[0]) + 0.5) * L / shape[0],
        (np.arange(shape[1]) + 0.5) * W / shape[1],
        (np.arange(shape[2]) + 0.5) * H / shape[2], indexing="ij"))
    field = np.sqrt(u * u + v * v + w * w)
    for _ in range(2):
        field = _box_blur(field, np.ones_like(field, dtype=bool))
    return field, np.asarray(arrivals), visit, p, age


def _box_blur(f, mask):
    """3-wide separable box blur of ``f``, renormalised at the mask edges."""
    w = mask.astype(float)
    out = np.zeros_like(f)
    for axis in range(3):
        fs = np.zeros_like(f)
        ws = np.zeros_like(w)
        for d in (-1, 0, 1):
            fs += np.roll(f * w, d, axis=axis)
            ws += np.roll(w, d, axis=axis)
        f = np.where(ws > 0, fs / np.maximum(ws, 1e-9), f)
        out = f
    return out * (w > 0)


def flow_ramp(tmax):
    """Speed: still (black) -> deep blue -> cyan -> green -> gold -> red."""
    stops = [(0.0, (0.0, 0.0, 0.0)), (0.06, (0.0, 0.0, 0.0)),
             (0.10, (0.05, 0.12, 0.45)), (0.30, (0.05, 0.45, 0.85)),
             (0.50, (0.10, 0.85, 0.75)), (0.68, (0.55, 0.95, 0.20)),
             (0.84, (1.00, 0.80, 0.10)), (1.00, (1.00, 0.25, 0.05))]
    out = []
    for t in np.linspace(0.0, 1.0, 128):
        for (a0, c0), (a1, c1) in zip(stops, stops[1:]):
            if a0 <= t <= a1:
                u = (t - a0) / (a1 - a0) if a1 > a0 else 0.0
                out.append((t * tmax,) + tuple(c0[k] + u * (c1[k] - c0[k])
                                              for k in range(3)))
                break
    return out


def residence_ramp(tmax):
    """Diverging: cyan (young, short-circuiting) -> nothing -> orange (old).

    The transparent middle is the point.  A monotonic ramp fills the tank with
    one even haze; this leaves the middle-aged bulk invisible and shows only
    the two populations a contact tank is judged on.
    """
    stops = [(0.0, (0.0, 0.0, 0.0)), (0.06, (0.0, 0.0, 0.0)),
             (0.10, (0.05, 0.12, 0.45)), (0.30, (0.05, 0.45, 0.85)),
             (0.50, (0.10, 0.85, 0.75)), (0.68, (0.55, 0.95, 0.20)),
             (0.84, (1.00, 0.80, 0.10)), (1.00, (1.00, 0.25, 0.05))]
    out = []
    for t in np.linspace(0.0, 1.0, 128):
        for (a0, c0), (a1, c1) in zip(stops, stops[1:]):
            if a0 <= t <= a1:
                u = (t - a0) / (a1 - a0) if a1 > a0 else 0.0
                out.append((t * tmax,) + tuple(c0[k] + u * (c1[k] - c0[k])
                                              for k in range(3)))
                break
    return out


def tank_edges():
    """Thin grey outline of the tank and baffles, for context."""
    segs = [((0, 0, 0), (L, 0, 0)), ((L, 0, 0), (L, W, 0)),
            ((L, W, 0), (0, W, 0)), ((0, W, 0), (0, 0, 0)),
            ((0, 0, H), (L, 0, H)), ((L, 0, H), (L, W, H)),
            ((L, W, H), (0, W, H)), ((0, W, H), (0, 0, H)),
            ((0, 0, 0), (0, 0, H)), ((L, 0, 0), (L, 0, H)),
            ((L, W, 0), (L, W, H)), ((0, W, 0), (0, W, H))]
    for bx, side in BAFFLES:
        y0, y1 = (0.0, 2.1) if side > 0 else (W - 2.1, W)
        segs += [((bx, y0, 0), (bx, y0, H)), ((bx, y1, 0), (bx, y1, H)),
                 ((bx, y0, H), (bx, y1, H)), ((bx, y0, 0), (bx, y1, 0))]
    pts, lines = vtk.vtkPoints(), vtk.vtkCellArray()
    for a, b in segs:
        idl = vtk.vtkIdList()
        idl.InsertNextId(pts.InsertNextPoint(*a))
        idl.InsertNextId(pts.InsertNextPoint(*b))
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


def inlet_rake(ny=16, nz=5, jitter=0.06, seed=3):
    rng = np.random.default_rng(seed)
    pts = vtk.vtkPoints()
    for yy in np.linspace(0.35, W - 0.35, ny):
        for zz in np.linspace(0.20 * H, 0.80 * H, nz):
            pts.InsertNextPoint(0.7 + rng.normal(0, jitter),
                               yy + rng.normal(0, jitter),
                               zz + rng.normal(0, jitter * 0.5))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    return pd


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", type=int, default=8000)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--diffusivity", type=float, default=0.015)
    p.add_argument("--n-seeds-y", type=int, default=16)
    p.add_argument("--n-seeds-z", type=int, default=5)
    p.add_argument("--tube-radius", type=float, default=0.022,
                   help="streamline tubes; deliberately thin (they are context)")
    p.add_argument("--streamline-opacity", type=float, default=0.45)
    p.add_argument("--no-streamlines", action="store_true")
    p.add_argument("--no-particles", action="store_true")
    p.add_argument("--png", default="docs/contact_tank.png")
    p.add_argument("--size", nargs=2, type=int, default=[1400, 900])
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    field, arrivals, _, pos, ages = monte_carlo(
        n=args.particles, n_steps=args.steps, kappa=args.diffusivity)
    resid = field[field > 0]
    print(f"Monte Carlo: {args.particles} particles, {args.steps} steps, "
          f"kappa {args.diffusivity}")
    print(f"  visited {len(resid)}/{field.size} cells "
          f"({100*len(resid)/field.size:.0f}%)")
    if arrivals.size:
        a = np.sort(arrivals)
        print(f"  residence time of {len(a)} that left: "
              f"min {a.min():.2f}  p10 {a[len(a)//10]:.2f}  "
              f"median {np.median(a):.2f}  p90 {a[9*len(a)//10]:.2f}  "
              f"max {a.max():.2f}")
        print("  (a good contact tank has a narrow distribution; a short "
              "p10 means short-circuiting)")
    tmax = float(field.max()) or 1.0
    agemax = float(ages.max()) or 1.0

    actors = []

    # 1. The volume: mean residence time.  This is the layer that shows the flow.
    vol = VolumeActor([field], spacing=(L / field.shape[0], W / field.shape[1],
                                       H / field.shape[2]),
                      color_points=flow_ramp(tmax),
                      # Floor high enough that the stagnant pools stay clear
                      # and the volume reads as the jet, not as a filled box.
                      opacity_points=[(0.0, 0.0), (0.26 * tmax, 0.0),
                                      (0.36 * tmax, 0.32), (0.72 * tmax, 0.70),
                                      (tmax, 0.95)],
                      sample_distance=0.25)
    vol.prop.SetAmbient(0.60)          # mostly emissive: the colour is the data
    vol.prop.SetDiffuse(0.45)
    vol.prop.SetSpecular(0.25)
    vol.prop.SetSpecularPower(30)
    actors.append(vol.actor)

    # 2. Streamlines, understated -- shape only.
    if not args.no_streamlines:
        grid, spacing = flow_grid()
        sl = StreamlineActor([(u, v, w) for u, v, w in [grid]], spacing=spacing,
                             seeds=inlet_rake(args.n_seeds_y, args.n_seeds_z),
                             direction="both", max_propagation=46.0,
                             initial_step=0.15, max_steps=4000,
                             color_by_magnitude=False,
                             tube_radius=args.tube_radius, line_width=1.0)
        sp = sl.actor.GetProperty()
        sp.SetColor(0.75, 0.82, 0.90)   # neutral: the volume owns the colour
        sp.SetOpacity(args.streamline_opacity)
        sp.SetAmbient(0.7)
        sp.SetDiffuse(0.3)
        actors.append(sl.actor)

    # 3. The particles themselves, coloured by how long they had been in there.
    if not args.no_particles:
        rng = np.random.default_rng(1)
        idx = rng.choice(len(pos), size=min(180, len(pos)), replace=False)
        ramp = residence_ramp(agemax)
        cols = [ramp[min(int(ages[i] / agemax * 127), 127)][1:] for i in idx]
        actors.append(point_actor([tuple(pos[i]) for i in idx],
                                  color=[tuple(c) for c in cols],
                                  size=0.025, alpha=1.0))

    actors.append(tank_edges())

    cam = vtk.vtkCamera()
    cam.SetFocalPoint(0.5 * L, 0.5 * W, 0.45 * H)
    cam.SetPosition(0.5 * L + 1.1 * L, -0.85 * L, 0.55 * L)
    cam.SetViewUp(0, 0, 1)

    if args.interactive:
        viewer = Viewer4D(size=(args.size[0], args.size[1]),
                          bg_color=(0.01, 0.01, 0.02))
        viewer.ren_win.SetMultiSamples(8)
        viewer.ren.SetUseDepthPeeling(1)
        viewer.ren.SetMaximumNumberOfPeels(24)
        viewer.ren_win.SetAlphaBitPlanes(1)
        for a in actors:
            viewer.add_actor(a)
        viewpoint(viewer.ren, cam)
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
        viewer.start()
        return 0

    render_to_png(actors, args.png, size=(args.size[0], args.size[1]),
                  camera=cam, bg_color=(0.01, 0.01, 0.02), zoom=1.3)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
