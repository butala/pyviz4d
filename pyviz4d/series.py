import numpy as np
import vtk
from .viz import TemporalActor
from .io import parse_pvd


class ImageDataSeriesActor(TemporalActor):
    """
    Base TemporalActor that lazy-loads 3D image data (.vti) from a .pvd series on disk frame by frame.
    """
    def __init__(self, pvd_path: str):
        self.entries = parse_pvd(pvd_path)
        if not self.entries:
            raise ValueError(f"No datasets found in .pvd file: {pvd_path}")

        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(self.entries[0][1])
        self.reader.Update()

        self.current_idx = 0
        self.num_frames = len(self.entries)
        # Derived classes will attach their filter/mapper to self.reader.GetOutputPort()

    def get_output_port(self):
        return self.reader.GetOutputPort()

    def _update_file(self, current_time: float):
        frame_idx = int(current_time)
        clamped_idx = min(max(frame_idx, 0), self.num_frames - 1)
        if clamped_idx != self.current_idx:
            self.reader.SetFileName(self.entries[clamped_idx][1])
            self.reader.Modified()
            self.reader.Update()
            self.current_idx = clamped_idx


class IsosurfaceSeriesActor(ImageDataSeriesActor):
    """
    Isosurface contour actor backed by a .pvd time-series on disk.
    Memory-efficient: reads each binary .vti frame on demand.
    """
    def __init__(self, pvd_path: str, iso_values=None, colors=None, opacity=0.4):
        super().__init__(pvd_path)

        if iso_values is None:
            iso_values = [0.2, 0.5, 1.0, 2.0]

        self.contour = vtk.vtkFlyingEdges3D()
        self.contour.SetInputConnection(self.get_output_port())
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

        self.actor = actor

    def update(self, current_time: float):
        self._update_file(current_time)


class PolyDataSeriesActor(TemporalActor):
    """
    Actor that lazy-loads polygonal/particle/streamline data (.vtp) from a .pvd series on disk.
    """
    def __init__(self, pvd_path: str, color=(1.0, 1.0, 1.0), line_width=1.0, point_size=1.0):
        self.entries = parse_pvd(pvd_path)
        if not self.entries:
            raise ValueError(f"No datasets found in .pvd file: {pvd_path}")

        self.reader = vtk.vtkXMLPolyDataReader()
        self.reader.SetFileName(self.entries[0][1])
        self.reader.Update()

        self.current_idx = 0
        self.num_frames = len(self.entries)

        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(self.mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(line_width)
        actor.GetProperty().SetPointSize(point_size)

        super().__init__(actor)

    def update(self, current_time: float):
        frame_idx = int(current_time)
        clamped_idx = min(max(frame_idx, 0), self.num_frames - 1)
        if clamped_idx != self.current_idx:
            self.reader.SetFileName(self.entries[clamped_idx][1])
            self.reader.Modified()
            self.reader.Update()
            self.current_idx = clamped_idx
