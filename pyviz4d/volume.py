import numpy as np
import vtk
from vtk.util import numpy_support

from .viz import TemporalActor


def float_scalars(array: np.ndarray) -> vtk.vtkDataArray:
    """3-D (X, Y, Z) array -> a flat Fortran-ordered vtkFloatArray.

    Fortran order so VTK's x-fastest point indexing matches numpy's (X, Y, Z)
    shape.  This is the packing every frame swap in this module and in
    streamline.vector_field_to_vtk needs, so it lives here once.
    """
    flat = np.ascontiguousarray(array.flatten(order='F'), dtype=np.float32)
    return numpy_support.numpy_to_vtk(num_array=flat, deep=True,
                                      array_type=vtk.VTK_FLOAT)


def create_vtk_image_from_numpy(array: np.ndarray, spacing=(1.0, 1.0, 1.0),
                                origin=(0.0, 0.0, 0.0)) -> vtk.vtkImageData:
    """Converts a 3D numpy array of shape (X, Y, Z) into a vtkImageData object."""
    if array.ndim != 3:
        raise ValueError(f"array must be 3-D, got shape {array.shape}")

    image = vtk.vtkImageData()
    image.SetDimensions(array.shape)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    image.GetPointData().SetScalars(float_scalars(array))
    return image


def swap_scalars(image: vtk.vtkImageData, frames: list, current_idx: int,
                 current_time: float) -> int:
    """Point ``image`` at the frame for ``current_time``; return the new index.

    The numpy -> VTK copy is the expensive part of a frame swap, so nothing is
    re-packed when the frame has not changed.  ``current_idx`` is -1 until the
    first call, which forces frame 0 to be packed.
    """
    frame_idx = min(max(int(current_time), 0), len(frames) - 1)
    if frame_idx == current_idx:
        return current_idx
    image.GetPointData().SetScalars(float_scalars(frames[frame_idx]))
    image.Modified()
    return frame_idx


def matplotlib_ctf(cmap_name: str, scalar_min: float, scalar_max: float, n: int = 64):
    """Build a vtkColorTransferFunction sampling a Matplotlib colormap.

    Goes through :mod:`matplotlib.colormaps` rather than ``pyplot.get_cmap``:
    the colormap registry is all this needs, and pyplot drags in a GUI backend.
    """
    from matplotlib import colormaps
    try:
        cmap = colormaps[cmap_name]
    except KeyError:
        # Fall back to a safe, known-good scientific colormap.
        cmap = colormaps["viridis"]

    ctf = vtk.vtkColorTransferFunction()
    span = scalar_max - scalar_min
    if span <= 0:
        span = 1.0
    for i in range(n + 1):
        t = i / n
        r, g, b, _ = cmap(t)
        ctf.AddRGBPoint(scalar_min + t * span, r, g, b)
    return ctf


def power_opacity(scalar_min: float, scalar_max: float, power: float = 1.0,
                  max_opacity: float = 0.9, n: int = 64):
    """
    Build a vtkPiecewiseFunction that ramps from 0 at scalar_min to max_opacity
    at scalar_max using ``(t ** power)``. power > 1 keeps low values transparent
    and reserves opacity for the strongest signal (useful for sonar backscatter).
    """
    pwf = vtk.vtkPiecewiseFunction()
    span = scalar_max - scalar_min
    if span <= 0:
        span = 1.0
    for i in range(n + 1):
        t = i / n
        v = scalar_min + t * span
        pwf.AddPoint(v, max_opacity * (t ** power))
    return pwf


def contour_actor(source, iso_values=None, colors=None, opacity=0.4):
    """Flying-edges isosurface actor, returned as ``(contour, mapper, actor)``.

    ``source`` is a ``vtkImageData`` or any ``vtkAlgorithm`` producing one --
    a numpy-backed image for :class:`IsosurfaceActor`, a ``.vti`` reader for
    :class:`~pyviz4d.series.IsosurfaceSeriesActor`.  Both used to carry this
    pipeline verbatim.

    ``colors`` maps ``iso_values`` to RGB triples; the default LUT runs from
    dark purple to yellow, i.e. the two ends of Matplotlib's magma.
    """
    if iso_values is None:
        iso_values = [0.2, 0.5, 1.0, 2.0]

    contour = vtk.vtkFlyingEdges3D()
    if hasattr(source, "GetOutputPort"):
        contour.SetInputConnection(source.GetOutputPort())
    else:
        contour.SetInputData(source)
    contour.ComputeNormalsOn()
    contour.ComputeScalarsOn()
    for i, val in enumerate(iso_values):
        contour.SetValue(i, val)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(contour.GetOutputPort())
    mapper.SetScalarRange(min(iso_values), max(iso_values))

    lut = vtk.vtkColorTransferFunction()
    if colors is not None:
        for val, color in zip(iso_values, colors):
            lut.AddRGBPoint(val, *color)
    else:
        lut.AddRGBPoint(iso_values[0], 0.267, 0.004, 0.329)
        lut.AddRGBPoint(iso_values[-1], 0.993, 0.906, 0.144)
    mapper.SetLookupTable(lut)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)

    prop = actor.GetProperty()
    prop.SetOpacity(opacity)
    prop.SetSpecular(0.8)
    prop.SetSpecularPower(60)
    prop.SetDiffuse(0.7)
    prop.SetAmbient(0.2)
    return contour, mapper, actor


class VolumeActor(TemporalActor):
    """
    Volume rendering actor for time-series 3D density grids.
    Supports a list of 3D numpy arrays, mapping them dynamically over time.
    """
    def __init__(self, frames_density: list, spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0), color_points=None, opacity_points=None):
        self.frames_density = frames_density
        self.spacing = spacing
        self.origin = origin
        self.num_frames = len(frames_density)

        self.image = create_vtk_image_from_numpy(self.frames_density[0], self.spacing, self.origin)

        self.mapper = vtk.vtkSmartVolumeMapper()
        self.mapper.SetInputData(self.image)
        self.mapper.SetBlendModeToComposite()

        self.prop = vtk.vtkVolumeProperty()
        self.prop.ShadeOn()
        self.prop.SetInterpolationTypeToLinear()

        color = vtk.vtkColorTransferFunction()
        if color_points is None:
            color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color.AddRGBPoint(1.0, 1.0, 1.0, 1.0)
        else:
            for pt in color_points:
                color.AddRGBPoint(*pt)
        self.prop.SetColor(color)

        opacity = vtk.vtkPiecewiseFunction()
        if opacity_points is None:
            opacity.AddPoint(0.0, 0.0)
            opacity.AddPoint(1.0, 0.8)
        else:
            for pt in opacity_points:
                opacity.AddPoint(*pt)
        self.prop.SetScalarOpacity(opacity)

        volume = vtk.vtkVolume()
        volume.SetMapper(self.mapper)
        volume.SetProperty(self.prop)

        super().__init__(volume)
        self.current_idx = -1

    def update(self, current_time: float):
        self.current_idx = swap_scalars(self.image, self.frames_density,
                                        self.current_idx, current_time)


class IsosurfaceActor(TemporalActor):
    """
    Extracts striking, transparent geometric isosurfaces (contours) from 3D density grids.
    Great for high-quality scientific visualizations a-la Kitware/ParaView.
    """
    def __init__(self, frames_density: list, spacing=(1.0, 1.0, 1.0),
                 origin=(0.0, 0.0, 0.0), iso_values=None, colors=None, opacity=0.4):
        self.frames_density = frames_density
        self.spacing = spacing
        self.origin = origin
        self.num_frames = len(frames_density)

        self.image = create_vtk_image_from_numpy(self.frames_density[0], self.spacing, self.origin)
        self.contour, self.mapper, actor = contour_actor(
            self.image, iso_values=iso_values, colors=colors, opacity=opacity)

        super().__init__(actor)
        self.current_idx = -1

    def update(self, current_time: float):
        self.current_idx = swap_scalars(self.image, self.frames_density,
                                        self.current_idx, current_time)
