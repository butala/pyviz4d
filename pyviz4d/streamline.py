"""Streamline and centerline helpers for PyViz4D.

These utilities mirror the techniques described in the 2024 IEEE SciVis
contest-winning PlumeViz system:

  * ``extract_centerline``  -> per-slice maximum-intensity backbone (Figure 3, green curve)
  * ``gradient_field``      -> central-difference gradient of backscatter (Figure 4)
  * ``StreamlineActor``     -> 4th-order Runge-Kutta stream tracing (Figures 4 & 5)
"""

import numpy as np
import vtk
from vtk.util import numpy_support
from .viz import TemporalActor


def gradient_field(volume: np.ndarray, spacing=(1.0, 1.0, 1.0)):
    """Central-difference gradient of a 3-D (X, Y, Z) scalar volume.

    Returns three arrays ``(gx, gy, gz)`` in physical units of
    ``volume_unit / metre``.
    """
    gx, gy, gz = np.gradient(volume, *spacing)
    return gx, gy, gz


def vector_field_to_vtk(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray,
                        spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
                        array_name: str = "vectors"):
    """Pack three (X, Y, Z) component arrays into a vtkImageData vector field."""
    assert gx.shape == gy.shape == gz.shape, "component shapes must match"
    nx, ny, nz = gx.shape

    image = vtk.vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)

    flat = np.empty((nx * ny * nz, 3), dtype=np.float32)
    # Fortran-order flatten -> x varies fastest, matching vtkImageData ordering.
    flat[:, 0] = np.ascontiguousarray(gx.flatten(order="F"))
    flat[:, 1] = np.ascontiguousarray(gy.flatten(order="F"))
    flat[:, 2] = np.ascontiguousarray(gz.flatten(order="F"))

    arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
    arr.SetName(array_name)
    image.GetPointData().SetVectors(arr)
    return image


def extract_centerline(volume: np.ndarray, spacing=(1.0, 1.0, 1.0),
                       origin=(0.0, 0.0, 0.0), threshold=None,
                       smooth_window: int = 5):
    """Extract the maximum-intensity backbone of a plume.

    For every z-slice, the (x, y) location of the maximum value is taken as the
    plume centre. Slices whose peak falls below ``threshold`` (defaults to the
    median of positive values) are dropped, and the remaining curve is lightly
    smoothed along z.

    Returns ``(points, values)``: ``points`` is an (N, 3) array of physical
    coordinates and ``values`` the backscatter value at each point.
    """
    nx, ny, nz = volume.shape
    if threshold is None:
        positive = volume[volume > 0]
        threshold = float(np.median(positive)) if positive.size else 0.0

    pts, vals = [], []
    for k in range(nz):
        sl = volume[:, :, k]
        idx = int(np.argmax(sl))
        ix, iy = np.unravel_index(idx, (nx, ny))
        v = float(sl[ix, iy])
        if v < threshold:
            continue
        pts.append((origin[0] + ix * spacing[0],
                    origin[1] + iy * spacing[1],
                    origin[2] + k * spacing[2]))
        vals.append(v)

    pts = np.asarray(pts, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)

    if len(pts) >= 3 and smooth_window > 1:
        half = smooth_window // 2
        smoothed = pts.copy()
        for d in range(3):
            padded = np.pad(pts[:, d], half, mode="edge")
            kernel = np.ones(smooth_window) / smooth_window
            smoothed[:, d] = np.convolve(padded, kernel, mode="valid")
        pts = smoothed

    return pts, vals


def polydata_from_points(points: np.ndarray, closed: bool = False):
    """Turn an (N, 3) array into a vtkPolyData polyline."""
    pd = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    for p in points:
        vpts.InsertNextPoint(*p)
    pd.SetPoints(vpts)

    lines = vtk.vtkCellArray()
    n = len(points)
    lines.InsertNextCell(n + (1 if closed else 0))
    for i in range(n):
        lines.InsertCellPoint(i)
    if closed and n:
        lines.InsertCellPoint(0)
    pd.SetLines(lines)
    return pd


def centerline_seeds(volume: np.ndarray, spacing=(1.0, 1.0, 1.0),
                     origin=(0.0, 0.0, 0.0), threshold=None,
                     radial_jitter: float = 0.0, n_radial: int = 1,
                     seed: int = 0):
    """Seed points placed along the plume centreline.

    ``n_radial > 1`` places ``n_radial`` seeds in a small circle around each
    centreline point, which produces the radial (centre -> periphery) streamlines
    shown in Figure 4 of the PlumeViz paper.
    """
    pts, _ = extract_centerline(volume, spacing, origin, threshold=threshold)
    rng = np.random.default_rng(seed)

    seed_points = vtk.vtkPoints()
    for p in pts:
        if n_radial <= 1:
            seed_points.InsertNextPoint(*p)
            continue
        for a in np.linspace(0.0, 2.0 * np.pi, n_radial, endpoint=False):
            seed_points.InsertNextPoint(p[0] + radial_jitter * np.cos(a),
                                        p[1] + radial_jitter * np.sin(a),
                                        p[2])

    pd = vtk.vtkPolyData()
    pd.SetPoints(seed_points)
    return pd


def trace_streamlines(vector_field: vtk.vtkImageData, seeds: vtk.vtkPolyData,
                      direction: str = "both", max_propagation: float = 100.0,
                      initial_step: float = 0.05, max_steps: int = 2000,
                      terminal_speed: float = 1e-12):
    """Trace streamlines through ``vector_field`` with 4th-order Runge-Kutta.

    ``direction`` is one of "forward", "backward" or "both".
    """
    tracer = vtk.vtkStreamTracer()
    tracer.SetInputDataObject(vector_field)
    tracer.SetSourceData(seeds)
    tracer.SetIntegrator(vtk.vtkRungeKutta4())

    if direction == "forward":
        tracer.SetIntegrationDirectionToForward()
    elif direction == "backward":
        tracer.SetIntegrationDirectionToBackward()
    else:
        tracer.SetIntegrationDirectionToBoth()

    tracer.SetMaximumPropagation(max_propagation)
    tracer.SetInitialIntegrationStep(initial_step)
    tracer.SetMinimumIntegrationStep(initial_step * 0.1)
    tracer.SetMaximumIntegrationStep(initial_step * 10.0)
    tracer.SetMaximumNumberOfSteps(max_steps)
    tracer.SetTerminalSpeed(terminal_speed)
    tracer.SetInterpolatorTypeToDataSetPointLocator()
    tracer.SetComputeVorticity(False)
    tracer.Update()
    return tracer.GetOutput()


class CenterlineActor(TemporalActor):
    """Green plume-centreline curve (Figure 3) that follows a time series of
    scalar volumes."""

    def __init__(self, frames_volume: list, spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0), threshold=None, smooth_window: int = 5,
                 color=(0.2, 1.0, 0.2), tube_radius: float = 0.2,
                 line_width: float = 2.0):
        self.frames_volume = frames_volume
        self.num_frames = len(frames_volume)
        self.spacing = spacing
        self.origin = origin
        self.threshold = threshold
        self.smooth_window = smooth_window
        self.current_idx = -1

        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(vtk.vtkPoints())
        self.poly.SetLines(vtk.vtkCellArray())

        if tube_radius > 0.0:
            self.tube = vtk.vtkTubeFilter()
            self.tube.SetInputData(self.poly)
            self.tube.SetRadius(tube_radius)
            self.tube.SetNumberOfSides(8)
            self.tube.CappingOn()
            self.mapper = vtk.vtkPolyDataMapper()
            self.mapper.SetInputConnection(self.tube.GetOutputPort())
        else:
            self.mapper = vtk.vtkPolyDataMapper()
            self.mapper.SetInputData(self.poly)

        actor = vtk.vtkActor()
        actor.SetMapper(self.mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetAmbient(0.6)
        actor.GetProperty().SetDiffuse(0.6)
        super().__init__(actor)

    def update(self, current_time: float):
        frame_idx = min(max(int(current_time), 0), self.num_frames - 1)
        if frame_idx == self.current_idx:
            return
        self.current_idx = frame_idx

        pts, _ = extract_centerline(self.frames_volume[frame_idx], self.spacing,
                                    self.origin, threshold=self.threshold,
                                    smooth_window=self.smooth_window)
        if len(pts) < 2:
            return

        vpts = vtk.vtkPoints()
        for p in pts:
            vpts.InsertNextPoint(*p)
        self.poly.SetPoints(vpts)

        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(pts))
        for i in range(len(pts)):
            lines.InsertCellPoint(i)
        self.poly.SetLines(lines)
        self.poly.Modified()


class StreamlineActor(TemporalActor):
    """Streamline actor for time-series vector fields.

    ``frames_vector`` is a list of ``(gx, gy, gz)`` tuples (one per timestep).
    Seeds may be provided once (static) or as a callable ``seeds_fn(frame_idx)``
    so they can follow a moving plume centreline.
    """

    def __init__(self, frames_vector: list, spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0), seeds=None, seeds_fn=None,
                 direction: str = "both", max_propagation: float = 100.0,
                 initial_step: float = 0.05, max_steps: int = 2000,
                 color=(1.0, 1.0, 1.0), color_by_magnitude: bool = True,
                 colormap: str = "plasma", magnitude_range=None,
                 tube_radius: float = 0.0, line_width: float = 1.5):
        self.frames_vector = frames_vector
        self.num_frames = len(frames_vector)
        self.spacing = spacing
        self.origin = origin
        self.seeds_fn = seeds_fn
        self.direction = direction
        self.max_propagation = max_propagation
        self.initial_step = initial_step
        self.max_steps = max_steps
        self.color_by_magnitude = color_by_magnitude
        self.magnitude_range = magnitude_range
        self.tube_radius = tube_radius
        self.line_width = line_width
        self.color = color
        self.current_idx = -1

        self.field = vector_field_to_vtk(*frames_vector[0], spacing=spacing, origin=origin)

        if seeds is None and seeds_fn is None:
            seeds = vtk.vtkPolyData()
            p = vtk.vtkPoints()
            p.InsertNextPoint(*origin)
            seeds.SetPoints(p)
        self._static_seeds = seeds

        self.tracer = vtk.vtkStreamTracer()
        self.tracer.SetInputDataObject(self.field)
        self.tracer.SetIntegrator(vtk.vtkRungeKutta4())
        self._apply_tracer_settings()
        self._set_seeds(0)

        self._build_pipeline(colormap)

        super().__init__(self.actor)

    def _apply_tracer_settings(self):
        t = self.tracer
        if self.direction == "forward":
            t.SetIntegrationDirectionToForward()
        elif self.direction == "backward":
            t.SetIntegrationDirectionToBackward()
        else:
            t.SetIntegrationDirectionToBoth()
        t.SetMaximumPropagation(self.max_propagation)
        t.SetInitialIntegrationStep(self.initial_step)
        t.SetMinimumIntegrationStep(self.initial_step * 0.1)
        t.SetMaximumIntegrationStep(self.initial_step * 10.0)
        t.SetMaximumNumberOfSteps(self.max_steps)
        t.SetTerminalSpeed(1e-12)
        t.SetInterpolatorTypeToDataSetPointLocator()
        t.SetComputeVorticity(False)

    def _set_seeds(self, frame_idx):
        if self.seeds_fn is not None:
            self.tracer.SetSourceData(self.seeds_fn(frame_idx))
        else:
            self.tracer.SetSourceData(self._static_seeds)

    def _build_pipeline(self, colormap):
        input_port = self.tracer.GetOutputPort()
        if self.tube_radius > 0.0 and not self.color_by_magnitude:
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(self.tracer.GetOutputPort())
            tube.SetRadius(self.tube_radius)
            tube.SetNumberOfSides(8)
            tube.CappingOn()
            input_port = tube.GetOutputPort()

        if self.color_by_magnitude:
            from .volume import matplotlib_ctf
            self.calc = vtk.vtkArrayCalculator()
            self.calc.SetInputConnection(input_port)
            self.calc.AddVectorArrayName("vectors")
            self.calc.SetFunction("mag(vectors)")
            self.calc.SetResultArrayName("streamline_mag")
            self.calc.Update()

            if self.magnitude_range is None:
                arr = self.calc.GetOutput().GetPointData().GetArray("streamline_mag")
                self.magnitude_range = (0.0, arr.GetRange()[1]) if arr else (0.0, 1.0)

            self.mapper = vtk.vtkPolyDataMapper()
            self.mapper.SetInputConnection(self.calc.GetOutputPort())
            self.mapper.SetScalarModeToUsePointFieldData()
            self.mapper.SelectColorArray("streamline_mag")
            self.mapper.SetScalarRange(*self.magnitude_range)
            self.mapper.SetLookupTable(matplotlib_ctf(colormap, *self.magnitude_range))
            self.mapper.ScalarVisibilityOn()
        else:
            self.mapper = vtk.vtkPolyDataMapper()
            self.mapper.SetInputConnection(input_port)
            self.mapper.ScalarVisibilityOff()

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        if not self.color_by_magnitude:
            self.actor.GetProperty().SetColor(*self.color)
        self.actor.GetProperty().SetLineWidth(self.line_width)
        self.actor.GetProperty().SetAmbient(0.5)
        self.actor.GetProperty().SetDiffuse(0.7)

    def update(self, current_time: float):
        frame_idx = min(max(int(current_time), 0), self.num_frames - 1)
        if frame_idx == self.current_idx:
            return
        self.current_idx = frame_idx

        gx, gy, gz = self.frames_vector[frame_idx]
        nx, ny, nz = gx.shape
        flat = np.empty((nx * ny * nz, 3), dtype=np.float32)
        flat[:, 0] = np.ascontiguousarray(gx.flatten(order="F"))
        flat[:, 1] = np.ascontiguousarray(gy.flatten(order="F"))
        flat[:, 2] = np.ascontiguousarray(gz.flatten(order="F"))
        arr = numpy_support.numpy_to_vtk(flat, deep=True, array_type=vtk.VTK_FLOAT)
        arr.SetName("vectors")
        self.field.GetPointData().SetVectors(arr)
        self.field.Modified()

        self._set_seeds(frame_idx)
        self.tracer.Modified()
        self.tracer.Update()
