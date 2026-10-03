"""4-D scientific spatio-temporal visualization on pure VTK.

The public names of the submodules are re-exported here so a caller can write
``from pyviz4d import line_actor``; the same names are reachable as
``pyviz4d.primitives.line_actor`` and so on.  See README "API at a glance" for
which module each lives in.
"""
from .cityjson import read_cityjson
from .io import VTKSeriesWriter, parse_pvd
from .primitives import (
    get_color,
    line_actor,
    line_source,
    point_actor,
    render_to_png,
    spherical_voxel_actor,
    viewpoint,
)
from .series import IsosurfaceSeriesActor, PolyDataSeriesActor
from .streamline import (
    CenterlineActor,
    StreamlineActor,
    apply_tracer_settings,
    centerline_seeds,
    extract_centerline,
    gradient_field,
    polydata_from_points,
    trace_streamlines,
    vector_field_to_vtk,
)
from .viz import EarthViewer4D, TemporalActor, Viewer4D
from .volume import (
    IsosurfaceActor,
    VolumeActor,
    contour_actor,
    create_vtk_image_from_numpy,
    matplotlib_ctf,
    power_opacity,
)

__all__ = [
    "CenterlineActor",
    "EarthViewer4D",
    "IsosurfaceActor",
    "IsosurfaceSeriesActor",
    "PolyDataSeriesActor",
    "StreamlineActor",
    "TemporalActor",
    "VTKSeriesWriter",
    "Viewer4D",
    "VolumeActor",
    "apply_tracer_settings",
    "centerline_seeds",
    "contour_actor",
    "create_vtk_image_from_numpy",
    "extract_centerline",
    "get_color",
    "gradient_field",
    "line_actor",
    "line_source",
    "matplotlib_ctf",
    "parse_pvd",
    "point_actor",
    "polydata_from_points",
    "power_opacity",
    "read_cityjson",
    "render_to_png",
    "spherical_voxel_actor",
    "trace_streamlines",
    "vector_field_to_vtk",
    "viewpoint",
]
