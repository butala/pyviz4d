"""Tests for the primitives ported from SphericalCT's ``vtk_primitives``.

The load-bearing tests here are the two convention ones
(``test_spherical_voxel_first_point_is_latitude`` and
``test_theta_is_latitude_not_colatitude``): a colatitude implementation passes
every other test in this file and fails only those.
"""

import hashlib
import importlib.util
import os

import numpy as np
import pytest
import vtk

import pyviz4d
from pyviz4d import (get_color, line_source, line_actor, point_actor,
                     spherical_voxel_actor, render_to_png)
from pyviz4d.spherical_grid import spherical_grid_actor

# SphericalCT's port of pyvizvtk, the source of truth for connectivity.  Used
# for a live differential check when the checkout is present.
SPHERICALCT_PORT = os.path.join(
    "/Users/butala/src/SphericalCT/src/sphericalct/vis/vtk_primitives.py")

# sha256 of the flattened (int64) cell connectivity for N_theta = N_phi = 10,
# captured from SphericalCT's vtk_primitives.py at commit 8acd1da.  Embedded so
# the connectivity oracle survives without that checkout.
CELLS_SHA_10x10 = "33eb0d0a2a6cb6fbb9bb28c0fc71afeca4d929d912546fd9161f118b576a4826"

# vtkPoints defaults to float32, so point comparisons carry ~1e-7 relative noise.
F32 = dict(rel=1e-6, abs=1e-6)
N_THETA = N_PHI = 10


# --------------------------------------------------------------------- helpers

def _polydata(actor):
    """The polydata behind an actor, through its mapper."""
    return actor.GetMapper().GetInput()


def _points(actor):
    pd = _polydata(actor)
    return np.array([pd.GetPoint(i) for i in range(pd.GetNumberOfPoints())])


def _cells(actor):
    pd = _polydata(actor)
    out = []
    for c in range(pd.GetNumberOfCells()):
        cell = pd.GetCell(c)
        out.append(tuple(cell.GetPointId(k)
                         for k in range(cell.GetNumberOfPoints())))
    return out


def _cells_digest(actor):
    return hashlib.sha256(
        np.array(_cells(actor), dtype=np.int64).tobytes()).hexdigest()


def _load_sphericalct_port():
    spec = importlib.util.spec_from_file_location("sc_vtk_primitives",
                                                  SPHERICALCT_PORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corner_points(actor, n_theta=N_THETA, n_phi=N_PHI):
    """The cell's eight corner vertices, in the port's cap/index order."""
    pd = _polydata(actor)
    k = n_theta * n_phi
    ids = [base + i * n_phi + j
           for base in (0, k)
           for i in (0, n_theta - 1)
           for j in (0, n_phi - 1)]
    return np.array([pd.GetPoint(i) for i in ids], dtype=float)


def _analytic_centre(r1, r2, theta1, theta2, phi1, phi2, colatitude=False):
    """Centroid of a cell's 8 corners under one angular convention."""
    unit = []
    for t in (theta1, theta2):
        for p in (phi1, phi2):
            if colatitude:
                unit.append((np.sin(t) * np.cos(p),
                             np.sin(t) * np.sin(p),
                             np.cos(t)))
            else:  # latitude: z = r sin(theta)
                unit.append((np.cos(t) * np.cos(p),
                             np.cos(t) * np.sin(p),
                             np.sin(t)))
    return (r1 + r2) / 2.0 * np.mean(unit, axis=0)


# ------------------------------------------------------------------- get_color

@pytest.mark.parametrize("name,expected", [
    ("red", (1.0, 0.0, 0.0)),
    ("lime", (0.0, 1.0, 0.0)),
    ("blue", (0.0, 0.0, 1.0)),
    ("cyan", (0.0, 1.0, 1.0)),
    ("yellow", (1.0, 1.0, 0.0)),
    ("black", (0.0, 0.0, 0.0)),
    ("white", (1.0, 1.0, 1.0)),
])
def test_get_color_names(name, expected):
    assert get_color(name) == pytest.approx(expected)


def test_get_color_returns_plain_tuple():
    c = get_color("cyan")
    assert isinstance(c, tuple) and len(c) == 3


# ------------------------------------------------------------------ line_source

def test_line_source_multi_pair_counts_and_rgba():
    xyz1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    xyz2 = [(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
    pd = line_source(xyz1, xyz2, color=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
                     alpha=[0.5, 1.0])
    assert pd.GetNumberOfPoints() == 4
    assert pd.GetNumberOfCells() == 2
    # second point of cell i is indexed at len(xyz1) + i
    assert _cells_cells(pd) == [(0, 2), (1, 3)]
    scalars = pd.GetCellData().GetScalars()
    assert scalars.GetNumberOfComponents() == 4
    assert tuple(scalars.GetTuple4(0)) == (255, 0, 0, 127)
    assert tuple(scalars.GetTuple4(1)) == (0, 255, 0, 255)


def test_line_source_bare_point_broadcast_is_a_fan():
    origin = (0.0, 0.0, 0.0)
    targets = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    pd = line_source(origin, targets)
    assert pd.GetNumberOfPoints() == 4          # 1 origin + 3 targets
    assert pd.GetNumberOfCells() == 3
    assert _cells_cells(pd) == [(0, 1), (0, 2), (0, 3)]


def _cells_cells(pd):
    out = []
    for c in range(pd.GetNumberOfCells()):
        cell = pd.GetCell(c)
        out.append(tuple(cell.GetPointId(k)
                         for k in range(cell.GetNumberOfPoints())))
    return out


def test_line_source_no_color_means_no_cell_scalars():
    pd = line_source([(0, 0, 0)], [(1, 1, 1)])
    assert pd.GetNumberOfCells() == 1
    assert pd.GetCellData().GetScalars() is None


# ------------------------------------------------------------------- line_actor

def _cell0_rgba(actor):
    return tuple(_polydata(actor).GetCellData().GetScalars().GetTuple4(0))


def test_line_actor_cells_and_default_color_is_cyan():
    # line_actor passes the colour through line_source's per-cell RGBA, so the
    # cyan default shows up as cell scalars (dyed by the mapper), not on the
    # actor property.
    actor = line_actor([(0, 0, 0)], [(1, 1, 1)])
    assert _polydata(actor).GetNumberOfCells() == 1
    assert _cell0_rgba(actor) == (0, 255, 255, 255)


def test_line_actor_honours_color_and_width():
    actor = line_actor([(0, 0, 0)], [(1, 1, 1)], color=get_color("red"),
                       alpha=0.5, width=4)
    assert _cell0_rgba(actor) == (255, 0, 0, 127)
    assert actor.GetProperty().GetLineWidth() == pytest.approx(4.0)


# ------------------------------------------------------------------ point_actor

def test_point_actor_glyph_counts_and_rgba():
    xyz = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    color = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    actor = point_actor(xyz, color=color, size=0.25,
                        phi_resolution=10, theta_resolution=10)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(0.25)
    sphere.SetPhiResolution(10)
    sphere.SetThetaResolution(10)
    sphere.Update()
    n_pts = sphere.GetOutput().GetNumberOfPoints()
    n_polys = sphere.GetOutput().GetNumberOfPolys()

    glyph = actor.GetMapper().GetInput()
    assert glyph.GetNumberOfPoints() == len(xyz) * n_pts
    assert glyph.GetNumberOfCells() == len(xyz) * n_polys

    # per-point RGBA lives on the input polydata (that is what the glyph colours)
    scalars = actor.GetMapper().GetInputAlgorithm().GetInput().GetPointData().GetScalars()
    assert scalars.GetNumberOfComponents() == 4
    assert tuple(scalars.GetTuple4(0)) == (255, 0, 0, 255)
    assert tuple(scalars.GetTuple4(2)) == (0, 0, 255, 255)

    # flat, unshaded markers
    assert actor.GetProperty().GetAmbient() == pytest.approx(1.0)
    assert actor.GetProperty().GetDiffuse() == pytest.approx(0.0)


def test_point_actor_default_color_is_yellow():
    actor = point_actor([(0.0, 0.0, 0.0)])
    scalars = actor.GetMapper().GetInputAlgorithm().GetInput().GetPointData().GetScalars()
    assert tuple(scalars.GetTuple4(0)) == (255, 255, 0, 255)


# ------------------------------------------------------- spherical_voxel_actor

def test_spherical_voxel_counts_and_connectivity_match_sphericalct():
    # (8, 15, X) is (theta1, theta2, phi1) in degrees, r1 = 1, r2 = 2,
    # phi2 = phi1 + 45; N defaults to 10 x 10.
    cases = [
        dict(r1=1.0, r2=2.0, theta1=np.radians(8), theta2=np.radians(15),
             phi1=np.radians(0), phi2=np.radians(45)),
        dict(r1=1.0, r2=2.0, theta1=np.radians(8), theta2=np.radians(15),
             phi1=np.radians(30), phi2=np.radians(75)),
    ]
    for kw in cases:
        actor = spherical_voxel_actor(**kw)
        assert _polydata(actor).GetNumberOfPoints() == 200
        assert _polydata(actor).GetNumberOfCells() == 198
        assert _cells_digest(actor) == CELLS_SHA_10x10


def test_spherical_voxel_reversed_edges_are_normalised():
    base = spherical_voxel_actor(1.0, 2.0, np.radians(8), np.radians(15),
                                 np.radians(0), np.radians(45))
    # reversed radius, theta and phi order -- same points and same connectivity
    flipped = spherical_voxel_actor(2.0, 1.0, np.radians(15), np.radians(8),
                                    np.radians(45), np.radians(0))
    assert np.allclose(_points(base), _points(flipped))
    assert _cells(base) == _cells(flipped)


@pytest.mark.parametrize("theta1,theta2,pole_row,sign", [
    (np.radians(80), np.radians(90), N_THETA - 1, +1.0),   # north pole cap
    (np.radians(-90), np.radians(-80), 0, -1.0),           # south pole cap
])
def test_spherical_voxel_poles_sit_on_the_axis(theta1, theta2, pole_row, sign):
    actor = spherical_voxel_actor(1.0, 2.0, theta1, theta2,
                                  np.radians(0), np.radians(60))
    pd = _polydata(actor)
    k = N_THETA * N_PHI
    # the pole ring (theta = +-90 deg) collapses to (0, 0, +-r) on both caps
    for base, r in ((0, 1.0), (k, 2.0)):
        for j in range(N_PHI):
            p = pd.GetPoint(base + pole_row * N_PHI + j)
            assert p[0] == pytest.approx(0.0, **F32)
            assert p[1] == pytest.approx(0.0, **F32)
            assert p[2] == pytest.approx(sign * r, **F32)


# --------------------------------------------------------------- CONVENTION

def test_spherical_voxel_first_point_is_latitude():
    """Point 0 is cap r1 at (i=0, j=0): r1 * (cos t1 cos p1, cos t1 sin p1, sin t1).

    A colatitude implementation would put ``cos t1`` on z and ``sin t1`` on the
    radial part, so this pins the convention at the level of the raw points.
    """
    r1, r2 = 2.0, 5.0
    t1, t2 = np.radians(5), np.radians(70)
    p1, p2 = np.radians(0), np.radians(70)
    p0 = _points(spherical_voxel_actor(r1, r2, t1, t2, p1, p2))[0]
    expected = r1 * np.array([np.cos(t1) * np.cos(p1),
                              np.cos(t1) * np.sin(p1),
                              np.sin(t1)])
    assert p0 == pytest.approx(expected, **F32)
    # not the colatitude point
    colatitude = r1 * np.array([np.sin(t1) * np.cos(p1),
                                np.sin(t1) * np.sin(p1),
                                np.cos(t1)])
    assert not np.allclose(p0, colatitude, atol=0.1)


def test_theta_is_latitude_not_colatitude():
    """A theta1 > 0 cell's centre lands on +z, at the LATITUDE centroid.

    This is the one test a colatitude implementation fails: every other test
    here (counts, connectivity, reversed edges) is convention-blind.
    """
    r1, r2 = 2.0, 5.0
    t1, t2 = np.radians(5), np.radians(70)     # theta1 > 0 -> northern cell
    p1, p2 = np.radians(0), np.radians(70)

    actor = spherical_voxel_actor(r1, r2, t1, t2, p1, p2)
    centre = _corner_points(actor).mean(axis=0)      # cell centre = 8-corner centroid
    lat = _analytic_centre(r1, r2, t1, t2, p1, p2, colatitude=False)
    col = _analytic_centre(r1, r2, t1, t2, p1, p2, colatitude=True)

    assert centre == pytest.approx(lat, **F32)
    assert centre[2] > 0.0
    # the colatitude centre is a *different* point, by a wide margin
    assert not np.allclose(centre, col, atol=0.1)
    assert abs(col[2] - centre[2]) > 0.5

    # The pinned reference value for this family: latitude gives z ~ +1.80,
    # sitting in the +z hemisphere, whereas colatitude drops it towards the
    # equator.  (The reference's exact originating arguments were not recorded;
    # the nearest round-argument case above reproduces it to ~7e-3.)
    assert centre == pytest.approx((1.5692, 1.0931, 1.799), abs=0.01)


# ------------------------------------------------------- spherical_grid fix

def test_spherical_grid_actor_is_latitude():
    """The spokes use latitude: for spoke i, arcsin(z1/r1) == theta_i.

    Colatitude would recover ``-theta_i`` here (the two grids are complements,
    so the *set* of z values is identical -- only the per-index mapping
    distinguishes them, which is exactly what a colatitude clash gets wrong).
    """
    r1, r2, n_theta, n_phi = 1.0, 2.0, 6, 8
    actor = spherical_grid_actor(r1, r2, n_theta, n_phi)
    pd = _polydata(actor)
    assert pd.GetNumberOfCells() == n_phi * (n_theta - 1)

    starts = np.array([pd.GetPoint(2 * i) for i in range(pd.GetNumberOfCells())])
    recovered = np.arcsin(starts[:, 2] / r1)
    expected_grid = np.linspace(-np.pi / 2, np.pi / 2, n_theta + 1)[1:-1]
    # code order is phi-major, theta-minor
    expected = np.tile(expected_grid, n_phi)
    assert recovered == pytest.approx(expected, **F32)
    assert recovered.min() < 0.0 < recovered.max()


# --------------------------------------------------------------- offscreen PNG

def test_render_to_png_is_non_blank(tmp_path):
    import imageio.v2 as imageio

    out = tmp_path / "scene.png"
    actors = [
        line_actor([(0, 0, 0), (0, 0, 0)], [(1, 1, 1), (-1, 1, 0)], width=3),
        point_actor([(1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0)], size=0.25,
                    color=[(1, 0, 0), (0, 1, 0), (0, 0, 1)]),
        spherical_voxel_actor(1.5, 2.5, np.radians(10), np.radians(60),
                              np.radians(0), np.radians(80)),
    ]
    path = render_to_png(actors, out, size=(400, 300))
    assert os.path.exists(path)

    img = imageio.imread(path)
    assert img.shape == (300, 400, 3)
    flat = img.reshape(-1, img.shape[-1]).astype(int)

    # measured, not eyeballed: many distinct colours and a real fraction of
    # pixels away from the (0.15, 0.15, 0.15) background.
    unique_colors = np.unique(flat, axis=0).shape[0]
    bg = int(round(0.15 * 255))
    non_background = np.mean(np.abs(flat - bg).max(axis=1) > 8)
    assert unique_colors > 20
    assert non_background > 0.02


def test_viewer_save_screenshot(tmp_path):
    import imageio.v2 as imageio
    from pyviz4d import Viewer4D

    viewer = Viewer4D(size=(320, 240))
    viewer.add_actor(line_actor([(0, 0, 0)], [(1, 1, 1)], width=2))
    viewer.add_actor(point_actor([(1.0, 1.0, 1.0)], size=0.2))
    viewer.add_actor(spherical_voxel_actor(1.5, 2.5, np.radians(10),
                                           np.radians(60), np.radians(0),
                                           np.radians(80)))

    out = tmp_path / "viewer.png"
    viewer.save_screenshot(out)
    assert os.path.exists(out)
    img = imageio.imread(out)
    assert img.shape[:2] == (240, 320)
    bg = int(round(0.15 * 255))
    non_background = np.mean(
        np.abs(img.reshape(-1, img.shape[-1]).astype(int) - bg).max(axis=1) > 8)
    assert non_background > 0.01


# ---------------------------------------------------- live differential check

@pytest.mark.skipif(not os.path.exists(SPHERICALCT_PORT),
                    reason="SphericalCT checkout not present")
def test_matches_sphericalct_port_bitexact():
    ref = _load_sphericalct_port()
    cases = [
        (1.0, 2.0, np.radians(8), np.radians(15), np.radians(0), np.radians(45)),
        (1.0, 2.0, np.radians(8), np.radians(15), np.radians(30), np.radians(75)),
        (2.0, 5.0, np.radians(-20), np.radians(40), np.radians(-100), np.radians(30)),
    ]
    for args in cases:
        mine = spherical_voxel_actor(*args)
        theirs = ref.spherical_voxel_actor(*args)
        assert np.array_equal(_points(mine), _points(theirs))
        assert _cells(mine) == _cells(theirs)
