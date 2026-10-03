import vtk

from .io import parse_pvd
from .volume import contour_actor
from .viz import TemporalActor


def _advance(reader, entries, current_idx, current_time):
    """Point a series ``reader`` at the frame for ``current_time``.

    Returns the new frame index, which equals ``current_idx`` when the frame
    has not changed -- re-reading a .vti/.vtp from disk is the expensive part
    of a series actor's update.  ``current_idx`` is -1 until the first call,
    which forces frame 0 to be read.
    """
    frame_idx = min(max(int(current_time), 0), len(entries) - 1)
    if frame_idx == current_idx:
        return current_idx
    reader.SetFileName(entries[frame_idx][1])
    reader.Modified()
    reader.Update()
    return frame_idx


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

        self.current_idx = -1
        self.num_frames = len(self.entries)
        # Derived classes attach their filter/mapper to self.reader.GetOutputPort()
        # and replace this with the real vtkProp (see contour_actor).
        super().__init__(None)

    def get_output_port(self):
        return self.reader.GetOutputPort()

    def update(self, current_time: float):
        self.current_idx = _advance(self.reader, self.entries,
                                    self.current_idx, current_time)


class IsosurfaceSeriesActor(ImageDataSeriesActor):
    """
    Isosurface contour actor backed by a .pvd time-series on disk.
    Memory-efficient: reads each binary .vti frame on demand.
    """
    def __init__(self, pvd_path: str, iso_values=None, colors=None, opacity=0.4):
        super().__init__(pvd_path)
        self.contour, self.mapper, actor = contour_actor(
            self.reader, iso_values=iso_values, colors=colors, opacity=opacity)
        self.actor = actor


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

        self.current_idx = -1
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
        self.current_idx = _advance(self.reader, self.entries,
                                    self.current_idx, current_time)
