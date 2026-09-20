import numpy as np
import vtk
from vtk.util import numpy_support
from .viz import TemporalActor

def create_vtk_image_from_numpy(array: np.ndarray, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Converts a 3D numpy array of shape (X, Y, Z) into a vtkImageData object.
    """
    assert array.ndim == 3, "Array must be 3D"

    # Flatten in Fortran order so VTK correctly interprets (X, Y, Z) ordering
    flat_array = np.ascontiguousarray(array.flatten(order='F'), dtype=np.float32)

    vtk_array = numpy_support.numpy_to_vtk(num_array=flat_array, deep=True, array_type=vtk.VTK_FLOAT)

    image = vtk.vtkImageData()
    image.SetDimensions(array.shape)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    image.GetPointData().SetScalars(vtk_array)
    return image


def matplotlib_ctf(cmap_name: str, scalar_min: float, scalar_max: float, n: int = 64):
    """Build a vtkColorTransferFunction sampling a Matplotlib colormap."""
    import matplotlib.pyplot as plt
    try:
        cmap = plt.get_cmap(cmap_name)
    except (ValueError, KeyError):
        # Fall back to a safe, known-good scientific colormap.
        cmap = plt.get_cmap("viridis")

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
        self.current_idx = 0

    def update(self, current_time: float):
        frame_idx = int(current_time)
        if frame_idx != self.current_idx:
            clamped_idx = min(max(frame_idx, 0), self.num_frames - 1)

            flat_array = np.ascontiguousarray(self.frames_density[clamped_idx].flatten(order='F'), dtype=np.float32)
            vtk_array = numpy_support.numpy_to_vtk(num_array=flat_array, deep=True, array_type=vtk.VTK_FLOAT)

            self.image.GetPointData().SetScalars(vtk_array)
            self.image.Modified()
            self.current_idx = frame_idx

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

        if iso_values is None:
            iso_values = [0.2, 0.5, 1.0, 2.0]

        self.image = create_vtk_image_from_numpy(self.frames_density[0], self.spacing, self.origin)

        self.contour = vtk.vtkFlyingEdges3D()
        self.contour.SetInputData(self.image)
        self.contour.ComputeNormalsOn()
        self.contour.ComputeScalarsOn()

        for i, val in enumerate(iso_values):
            self.contour.SetValue(i, val)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.contour.GetOutputPort())
        self.mapper.SetScalarRange(min(iso_values), max(iso_values))

        lut = vtk.vtkColorTransferFunction()
        if colors is not None:
            for val, color in zip(iso_values, colors):
                lut.AddRGBPoint(val, *color)
        else:
            lut.AddRGBPoint(iso_values[0], 0.267, 0.004, 0.329)
            lut.AddRGBPoint(iso_values[-1], 0.993, 0.906, 0.144)

        self.mapper.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.SetMapper(self.mapper)

        prop = actor.GetProperty()
        prop.SetOpacity(opacity)
        prop.SetSpecular(0.8)
        prop.SetSpecularPower(60)
        prop.SetDiffuse(0.7)
        prop.SetAmbient(0.2)

        super().__init__(actor)
        self.current_idx = 0

    def update(self, current_time: float):
        frame_idx = int(current_time)
        if frame_idx != self.current_idx:
            clamped_idx = min(max(frame_idx, 0), self.num_frames - 1)

            flat_array = np.ascontiguousarray(self.frames_density[clamped_idx].flatten(order='F'), dtype=np.float32)
            vtk_array = numpy_support.numpy_to_vtk(num_array=flat_array, deep=True, array_type=vtk.VTK_FLOAT)

            self.image.GetPointData().SetScalars(vtk_array)
            self.image.Modified()
            self.current_idx = frame_idx
