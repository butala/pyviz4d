"""A live, endless particle cloud advected through a velocity field.

This is the 4-D half of the package: not a precomputed time series, but a
simulation that runs.  See :class:`ParticleCloudActor`.
"""
import numpy as np
import vtk

from .viz import TemporalActor


def _turbulent_cmap(n=256):
    """turbo, as ``[(t, r, g, b), ...]`` with t in [0, 1]."""
    from matplotlib import colormaps
    ramp = colormaps["turbo"]
    return [(t,) + tuple(ramp(t)[:3]) for t in np.linspace(0.0, 1.0, n)]


class ParticleCloudActor(TemporalActor):
    """A Monte Carlo particle cloud that runs for ever.

    Every call to :meth:`update` integrates every particle one RK4 step
    through ``velocity_fn`` plus a Gaussian random increment -- the eddy
    diffusivity, which is what makes this a Monte Carlo rather than a
    streamline plot.  Any particle that leaves ``bounds`` is reborn in
    ``inlet`` with age zero, so the cloud reaches a statistical steady state
    and never runs out of particles or fills up: there is no last frame.

    Drawn as semi-transparent gaussian splats (:class:`vtkPointGaussianMapper`),
    so thousands of them accumulate into a cloud you can see through.

    Parameters
    ----------
    velocity_fn
        ``f(x, y, z) -> (u, v, w)``, evaluated on numpy arrays of any shape.
        Called four times per update (the RK4 stages) on ``(n,)`` arrays.
    bounds, inlet
        ``((x0, x1), (y0, y1), (z0, z1))``.  ``bounds`` is where particles live
        before being recycled; ``inlet`` is the box they are reborn in.
    n, dt, kappa
        Particle count, integration step, and the diffusivity of the random
        walk (variance ``2*kappa*dt`` per step).
    color_by
        ``"age"`` (time since respawn) or ``"speed"``.
    age_max
        The colour scale's upper end for ``color_by="age"``.  Defaults to the
        initial maximum, so the scale stays put as the cloud evolves.

    Notes
    -----
    Cost is one vectorised numpy pass over ``n`` particles per frame; 30k
    particles is a few milliseconds and runs at the frame rate of the window.
    """

    def __init__(self, velocity_fn, bounds, inlet, n=30000, dt=0.02,
                 kappa=0.015, color_by="age", cmap=None, age_max=None,
                 speed_max=None, splat_scale=1.0, opacity=0.30, seed=0):
        self.velocity_fn = velocity_fn
        self.bounds = tuple((float(a), float(b)) for a, b in bounds)
        self.inlet = tuple((float(a), float(b)) for a, b in inlet)
        self.n = int(n)
        self.dt = float(dt)
        self.kappa = float(kappa)
        self.color_by = color_by
        self.splat_scale = float(splat_scale)
        self.opacity = float(opacity)
        self.rng = np.random.default_rng(seed)

        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        self.pos = self.rng.uniform(lo, hi, size=(self.n, 3))
        # seeded with random ages so the cloud looks established from frame 1
        self.age = self.rng.uniform(0.0, 1.0, size=self.n)

        self.age_max = float(age_max) if age_max else 1.0
        self.speed_max = float(speed_max) if speed_max else 1.0
        self.cmap = cmap or _turbulent_cmap()
        self.steps = 0

        self.poly = vtk.vtkPolyData()
        self.points = vtk.vtkPoints()
        self.points.SetData(self._vtk_points())
        self.poly.SetPoints(self.points)
        verts = vtk.vtkCellArray()
        verts.InsertNextCell(self.n)
        for i in range(self.n):
            verts.InsertCellPoint(i)
        self.poly.SetVerts(verts)

        self.rgba = vtk.vtkUnsignedCharArray()
        self.rgba.SetNumberOfComponents(4)
        self.rgba.SetName("colors")
        self.poly.GetPointData().SetScalars(self.rgba)
        self._recolor()

        mapper = vtk.vtkPointGaussianMapper()
        mapper.SetInputData(self.poly)
        # World units, not pixels: the tank is 12 m across, so the splat
        # radius has to be centimetres.  (2.2 here drew one blob the size
        # of the domain.)
        mapper.SetScaleFactor(0.032 * self.splat_scale)
        mapper.SetColorModeToDirectScalars()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        super().__init__(actor)
        self.mapper = mapper

    # ---------------------------------------------------------------- physics

    def _vel(self, q):
        u, v, w = self.velocity_fn(q[:, 0], q[:, 1], q[:, 2])
        return np.column_stack([u, v, w]).astype(float)

    def _escaped(self, q):
        lo = np.array([b[0] for b in self.bounds])
        hi = np.array([b[1] for b in self.bounds])
        return np.any((q < lo) | (q > hi), axis=1)

    def _respawn(self, mask):
        lo = np.array([b[0] for b in self.inlet])
        hi = np.array([b[1] for b in self.inlet])
        k = int(mask.sum())
        self.pos[mask] = self.rng.uniform(lo, hi, size=(k, 3))
        self.age[mask] = 0.0

    def update(self, current_time: float):
        """Advance the whole cloud one step.

        ``current_time`` is ignored on purpose: a live simulation advances its
        own clock, one fixed ``dt`` per call, rather than jumping to whatever
        the playback slider says.
        """
        dt, q = self.dt, self.pos
        k1 = self._vel(q)
        k2 = self._vel(q + 0.5 * dt * k1)
        k3 = self._vel(q + 0.5 * dt * k2)
        k4 = self._vel(q + dt * k3)
        self.pos = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.pos += np.sqrt(2.0 * self.kappa * dt) * self.rng.normal(
            size=self.pos.shape)
        self.age += dt

        gone = self._escaped(self.pos)
        if gone.any():
            self._respawn(gone)
        self.steps += 1

        self.points.SetData(self._vtk_points())
        self.poly.SetPoints(self.points)
        self._recolor()
        self.poly.Modified()

    # ------------------------------------------------------------------- draw

    def _vtk_points(self):
        arr = vtk.vtkFloatArray()
        arr.SetNumberOfComponents(3)
        arr.SetNumberOfTuples(self.n)
        flat = np.ascontiguousarray(self.pos, dtype=np.float32)
        arr.SetVoidArray(flat.ravel(), flat.size, 1)
        # keep the buffer alive: SetVoidArray does not copy
        self._buf = flat
        pts = vtk.vtkPoints()
        pts.SetData(arr)
        return arr

    def _recolor(self):
        if self.color_by == "speed":
            sp = np.linalg.norm(self._vel(self.pos), axis=1)
            t = np.clip(sp / max(self.speed_max, 1e-9), 0.0, 1.0)
        else:
            t = np.clip(self.age / max(self.age_max, 1e-9), 0.0, 1.0)
        idx = (t * (len(self.cmap) - 1)).astype(int)
        a = int(max(0.0, min(1.0, self.opacity)) * 255)
        self.rgba.SetNumberOfTuples(self.n)
        for i in range(self.n):
            _, r, g, b = self.cmap[idx[i]]
            self.rgba.SetTuple4(i, int(r * 255), int(g * 255), int(b * 255), a)
        self.rgba.Modified()
