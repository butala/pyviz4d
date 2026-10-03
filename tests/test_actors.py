"""Frame-by-frame actors: volume, isosurface, series and streamlines.

These are the time-varying half of the package and had no coverage at all,
which is how the frame-swap logic ended up written out twice and the
``current_idx`` sentinel inconsistent between classes.
"""
import os

import numpy as np
import pytest
import vtk

from pyviz4d import (CenterlineActor, IsosurfaceActor, IsosurfaceSeriesActor,
                     PolyDataSeriesActor, StreamlineActor, VolumeActor,
                     VTKSeriesWriter, centerline_seeds, create_vtk_image_from_numpy,
                     gradient_field, line_source, matplotlib_ctf, polydata_from_points,
                     power_opacity, trace_streamlines, vector_field_to_vtk)
from pyviz4d.streamline import apply_tracer_settings
from pyviz4d.volume import contour_actor


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    return [rng.random((6, 5, 4)) + i for i in range(3)]


# ------------------------------------------------------- image-backed actors

@pytest.mark.parametrize("cls", [VolumeActor, IsosurfaceActor])
def test_update_swaps_frames_and_clamps(cls, frames):
    a = cls(frames)
    assert a.current_idx == -1                       # -1 forces frame 0

    a.update(0.0)
    assert a.current_idx == 0
    first = np.array(a.image.GetPointData().GetScalars()).copy()

    a.update(0.4)                                    # same frame: no re-pack
    assert a.current_idx == 0

    a.update(99.0)                                   # clamps to the last frame
    assert a.current_idx == 2
    assert not np.allclose(first, np.array(a.image.GetPointData().GetScalars()))


@pytest.mark.parametrize("cls", [VolumeActor, IsosurfaceActor])
def test_temporal_actor_exposes_a_vtk_prop(cls, frames):
    a = cls(frames)
    assert a.actor is not None


def test_contour_actor_accepts_an_image_or_a_reader(frames):
    img = create_vtk_image_from_numpy(frames[0])
    _, mapper, actor = contour_actor(img)
    assert isinstance(actor, vtk.vtkActor)
    assert mapper.GetLookupTable() is not None

    reader = vtk.vtkXMLImageDataReader()             # the series call site
    _, mapper2, actor2 = contour_actor(reader)
    assert isinstance(actor2, vtk.vtkActor)


def test_matplotlib_ctf_falls_back_on_an_unknown_name():
    assert isinstance(matplotlib_ctf("turbo", 0.0, 1.0),
                      vtk.vtkColorTransferFunction)
    assert isinstance(matplotlib_ctf("no-such-colormap", 0.0, 1.0),
                      vtk.vtkColorTransferFunction)


def test_create_vtk_image_rejects_a_2d_array():
    with pytest.raises(ValueError):
        create_vtk_image_from_numpy(np.zeros((2, 2)))


# --------------------------------------------------------------- series actors

@pytest.fixture
def series_dir(tmp_path, frames):
    w = VTKSeriesWriter(str(tmp_path), "s")
    for i, f in enumerate(frames):
        w.write_image_data(f, timestep=float(i))
    return tmp_path


def test_isosurface_series_actor_has_an_actor_and_advances(series_dir):
    a = IsosurfaceSeriesActor(os.path.join(str(series_dir), "s.pvd"),
                              iso_values=[1.0, 1.5])
    assert a.actor is not None
    a.update(0.0)
    assert a.current_idx == 0
    a.update(2.9)
    assert a.current_idx == 2


def test_polydata_series_actor_advances(tmp_path):
    w = VTKSeriesWriter(str(tmp_path), "p")
    for i in range(3):
        w.write_poly_data(polydata_from_points(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, float(i)]])), timestep=float(i))
    a = PolyDataSeriesActor(os.path.join(str(tmp_path), "p.pvd"))
    a.update(0.0)
    assert a.current_idx == 0
    a.update(1.2)
    assert a.current_idx == 1


def test_empty_pvd_is_an_error(tmp_path):
    (tmp_path / "empty.pvd").write_text(
        '<VTKFile type="Collection"><Collection></Collection></VTKFile>')
    with pytest.raises(ValueError):
        PolyDataSeriesActor(str(tmp_path / "empty.pvd"))


# ------------------------------------------------------------------ streamlines

def test_trace_streamlines_produces_cells(frames):
    gx, gy, gz = gradient_field(frames[0])
    out = trace_streamlines(vector_field_to_vtk(gx, gy, gz),
                            centerline_seeds(frames[0]))
    assert out.GetNumberOfCells() >= 1


@pytest.mark.parametrize("by_magnitude,tube", [(True, 0.0), (False, 0.0),
                                               (False, 0.1)])
def test_streamline_actor_paths_advance(frames, by_magnitude, tube):
    gx, gy, gz = gradient_field(frames[0])
    vecs = [(gx, gy, gz), (gx * 2, gy, gz)]
    a = StreamlineActor(vecs, seeds=centerline_seeds(frames[0]),
                        color_by_magnitude=by_magnitude, tube_radius=tube)
    a.update(0.0)
    assert a.current_idx == 0
    a.update(1.9)
    assert a.current_idx == 1


def test_centerline_actor_advances(frames):
    a = CenterlineActor([frames[0], frames[1]])
    a.update(0.0)
    assert a.current_idx == 0
    a.update(1.0)
    assert a.current_idx == 1


def test_vector_field_to_vtk_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        vector_field_to_vtk(np.zeros((2, 2, 2)), np.zeros((3, 3, 3)),
                            np.zeros((2, 2, 2)))


def test_apply_tracer_settings_is_shared():
    t1, t2 = vtk.vtkStreamTracer(), vtk.vtkStreamTracer()
    apply_tracer_settings(t1, direction="forward", max_propagation=5.0,
                          initial_step=0.2, max_steps=10)
    apply_tracer_settings(t2, direction="forward", max_propagation=5.0,
                          initial_step=0.2, max_steps=10)
    assert t1.GetMaximumPropagation() == t2.GetMaximumPropagation() == 5.0
    assert t1.GetMaximumNumberOfSteps() == t2.GetMaximumNumberOfSteps() == 10


# ------------------------------------------------------------------- primitives

def test_line_source_requires_alpha_with_color():
    with pytest.raises(ValueError):
        line_source([(0.0, 0.0, 0.0)], [(1.0, 1.0, 1.0)], color=(1.0, 0.0, 0.0))


def test_power_opacity_ramps_to_max():
    pwf = power_opacity(0.0, 1.0, power=2.0, max_opacity=0.9)
    assert pwf.GetSize() >= 2
