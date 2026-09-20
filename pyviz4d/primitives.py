"""Reusable VTK primitives: segment fans, point glyphs and solid spherical cells.

These are the five actors SphericalCT's visualisation layer needs, ported from
``sphericalct/vis/vtk_primitives.py`` at SphericalCT commit ``8acd1da``.  That
file is a port of pyvizvtk at revision ``0ef1897`` and was kept
behaviour-compatible (identical points and cell connectivity) so that existing
renderings do not shift; this module is a port of *that*, not of pyvizvtk
directly, so the ``line_actor`` default-colour fix and the ``width`` keyword
come along.

pyviz4d on its own only had ``streamline.polydata_from_points`` (one polyline,
not a segment list) and ``spherical_grid.spherical_grid_actor`` (grid spokes,
not solid cells), so the actors below are new here.

Conventions
-----------
Every angular helper in pyviz4d uses **latitude** in ``[-pi/2, +pi/2]`` with the
polar axis on ``z``::

    x = r cos(theta) cos(phi)
    y = r cos(theta) sin(phi)
    z = r sin(theta)

This matches :func:`pyviz4d.earth.sphere_to_cartesian` / ``wgs84_to_cartesian``
(lines 35-39 and 52-54) and SphericalCT's own latitude frame
(``src/coordinates.cpp:sph2cart``, ``UniformHollowSphere.theta_edges``).  It is
*not* colatitude (``z = r cos(theta)``): a colatitude helper with the same name
in the same package is the failure mode this package avoids.  See
``tests/test_primitives.py::test_theta_is_latitude_not_colatitude``.

Note that ``line_source`` is *not* ``vtkLineSource``: it takes explicit endpoint
lists and emits one two-point cell per ``(start, end)`` pair, so an entire ray
fan is a single polydata.
"""

from collections.abc import Iterable
from itertools import repeat

import matplotlib.colors
import numpy as np
import vtk


def get_color(name):
    """RGB tuple in [0, 1] for a matplotlib colour name (``'cyan'``, ``'red'``)."""
    return tuple(matplotlib.colors.to_rgb(name))


def line_source(xyz1, xyz2, color=None, alpha=None):
    """PolyData of straight segments from ``xyz1[i]`` to ``xyz2[i]``.

    ``xyz1`` and ``xyz2`` are sequences of points.  A bare point (not a
    sequence of points) is broadcast: ``line_source(origin, targets)`` is a
    fan.  One two-point cell is emitted per pair, with the second point indexed
    at ``len(xyz1) + j``.  When ``color`` is given, ``alpha`` must be too, and
    both may be scalars (repeated) or per-segment sequences; they are written as
    a per-cell ``vtkUnsignedCharArray`` with four components on
    ``GetCellData()``.
    """
    if isinstance(xyz1[0], Iterable):
        xyz1_id = list(range(len(xyz1)))
    else:
        xyz1 = [xyz1]
        xyz1_id = repeat(0)

    if isinstance(xyz2[0], Iterable):
        xyz2_id = list(range(len(xyz2)))
    else:
        xyz2 = [xyz2]
        xyz2_id = repeat(0)

    N = max(len(xyz1), len(xyz2))

    if color is not None:
        assert alpha is not None
        if not isinstance(color[0], Iterable):
            color = repeat(color, N)
        if not isinstance(alpha, Iterable):
            alpha = repeat(alpha, N)

    points = vtk.vtkPoints()
    for xyz1_i in xyz1:
        points.InsertNextPoint(xyz1_i)
    for xyz2_i in xyz2:
        points.InsertNextPoint(xyz2_i)

    lines = vtk.vtkCellArray()
    for i, j, _ in zip(xyz1_id, xyz2_id, range(N)):
        lines.InsertNextCell(2)
        lines.InsertCellPoint(i)
        lines.InsertCellPoint(len(xyz1) + j)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    if color is not None:
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(4)
        for color_i, alpha_i in zip(color, alpha):
            colors.InsertNextTuple4(int(color_i[0] * 255),
                                    int(color_i[1] * 255),
                                    int(color_i[2] * 255),
                                    int(alpha_i * 255))
        polydata.GetCellData().SetScalars(colors)

    return polydata


def line_actor(xyz1, xyz2, color=None, alpha=1, width=None):
    """Actor drawing the segments of :func:`line_source`.

    ``color=None`` means cyan, the pyvizvtk default.  ``width`` is the line
    width in pixels; ``None`` leaves VTK's default.
    """
    if color is None:
        color = get_color('cyan')

    polydata = line_source(xyz1, xyz2, color=color, alpha=alpha)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    if width is not None:
        actor.GetProperty().SetLineWidth(width)
    return actor


def point_actor(xyz, color=None, size=100, phi_resolution=10, theta_resolution=10,
                alpha=1):
    """Actor drawing small spheres at ``xyz`` via ``vtkGlyph3D``.

    ``size`` is the glyph sphere *radius* in world units, which is why callers
    pass values of order 0.01 next to coordinates of order 1.  ``color=None``
    means yellow, the pyvizvtk default.  The glyphs are drawn fully ambient so
    that they read as flat markers rather than shaded balls.  Per-point RGBA is
    written as a four-component ``vtkUnsignedCharArray`` on ``GetPointData()``
    and selected by ``SetColorModeToColorByScalar``.
    """
    if color is None:
        color = get_color('yellow')

    if not isinstance(xyz[0], Iterable):
        xyz = [xyz]

    N = len(xyz)

    if not isinstance(color[0], Iterable):
        color = repeat(color, N)

    if not isinstance(alpha, Iterable):
        alpha = repeat(alpha, N)

    points = vtk.vtkPoints()
    for xyz_i in xyz:
        points.InsertNextPoint(xyz_i)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(4)
    for color_i, alpha_i in zip(color, alpha):
        colors.InsertNextTuple4(int(color_i[0] * 255),
                                int(color_i[1] * 255),
                                int(color_i[2] * 255),
                                int(alpha_i * 255))
    polydata.GetPointData().SetScalars(colors)

    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(size)
    sphere.SetPhiResolution(phi_resolution)
    sphere.SetThetaResolution(theta_resolution)

    glyphs = vtk.vtkGlyph3D()
    glyphs.SetInputData(polydata)
    glyphs.SetSourceConnection(sphere.GetOutputPort())
    glyphs.SetColorModeToColorByScalar()
    glyphs.ScalingOff()
    glyphs.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(glyphs.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(1)
    actor.GetProperty().SetDiffuse(0)
    return actor


def spherical_voxel_actor(r1, r2, theta1, theta2, phi1, phi2,
                          N_r=10, N_theta=10, N_phi=10):
    """The closed solid cell of the spherical grid with the six given faces.

    ``theta`` is **latitude** in ``[-pi/2, +pi/2]`` (see the module docstring):
    the cell spans radii ``r1``-``r2``, latitudes ``theta1``-``theta2`` and
    longitudes ``phi1``-``phi2``.  The six faces are the two radial caps
    (``r = r1``, ``r = r2``) and the four side walls (``phi = phi1``,
    ``phi = phi2``, ``theta = theta1``, ``theta = theta2``), tessellated
    ``N_theta`` x ``N_phi`` and emitted as quads in a ``vtkPolyData``.  ``N_r``
    is accepted for pyvizvtk signature compatibility and is not used there
    either: the cell is drawn as a shell, not resampled radially.

    Point ``(i theta, j phi)`` on a cap is index ``i*N_phi + j``; the second cap
    is offset by ``K = N_theta*N_phi``.  The argument order is normalised, so
    unsorted ``r1``/``r2``, ``theta1``/``theta2`` or ``phi1``/``phi2`` are
    harmless (both the values *and* the cell corner ordering come out the same).
    """
    if r1 > r2:
        r1, r2 = r2, r1
    if phi1 > phi2:
        phi1, phi2 = phi2, phi1
    if theta1 > theta2:
        theta1, theta2 = theta2, theta1

    phi_vec = np.linspace(phi1, phi2, N_phi)
    theta_vec = np.linspace(theta1, theta2, N_theta)
    phi, theta = np.meshgrid(phi_vec, theta_vec)

    # LATITUDE: z = r sin(theta), x/y through cos(theta).  The cap point at
    # (i theta, j phi) is index i*N_phi + j; the second cap is offset by K.
    x_r = np.cos(theta) * np.cos(phi)
    y_r = np.cos(theta) * np.sin(phi)
    z_r = np.sin(theta)

    points = vtk.vtkPoints()
    for r in (r1, r2):
        for x_r_i, y_r_i, z_r_i in zip(x_r.flat, y_r.flat, z_r.flat):
            points.InsertNextPoint((r * x_r_i, r * y_r_i, r * z_r_i))

    K = N_theta * N_phi

    def cap(i, j, base):
        return base + i * N_phi + j

    cells = []
    # surface 1: r = r1
    for i in range(N_theta - 1):
        for j in range(N_phi - 1):
            cells.append((cap(i, j, 0), cap(i + 1, j, 0),
                          cap(i + 1, j + 1, 0), cap(i, j + 1, 0)))
    # surface 2: r = r2 -- same id order as surface 1, as in the original
    for i in range(N_theta - 1):
        for j in range(N_phi - 1):
            cells.append((cap(i, j, K), cap(i + 1, j, K),
                          cap(i + 1, j + 1, K), cap(i, j + 1, K)))
    # surface 3: phi = phi1
    for i in range(N_theta - 1):
        cells.append((cap(i, 0, 0), cap(i, 0, K),
                      cap(i + 1, 0, K), cap(i + 1, 0, 0)))
    # surface 4: phi = phi2
    for i in range(N_theta - 1):
        cells.append((cap(i, N_phi - 1, 0), cap(i + 1, N_phi - 1, 0),
                      cap(i + 1, N_phi - 1, K), cap(i, N_phi - 1, K)))
    # surface 5: theta = theta1
    for j in range(N_phi - 1):
        cells.append((cap(0, j, 0), cap(0, j, K),
                      cap(0, j + 1, K), cap(0, j + 1, 0)))
    # surface 6: theta = theta2
    for j in range(N_phi - 1):
        cells.append((cap(N_theta - 1, j, 0), cap(N_theta - 1, j + 1, 0),
                      cap(N_theta - 1, j + 1, K), cap(N_theta - 1, j, K)))

    polys = vtk.vtkCellArray()
    for cell in cells:
        polys.InsertNextCell(len(cell))
        for point_id in cell:
            polys.InsertCellPoint(point_id)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    return actor


def _render_window_to_png(ren_win, path, scale=1):
    """Render ``ren_win`` and write the back buffer to ``path`` as a PNG.

    Pure VTK: ``vtkWindowToImageFilter`` with ``SetScale``, RGB output and
    ``ReadFrontBufferOff`` (the front buffer may be absent or occluded when the
    window is offscreen), then ``vtkPNGWriter``.
    """
    ren_win.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(ren_win)
    w2i.SetScale(int(scale))
    w2i.SetInputBufferTypeToRGB()
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(path))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()
    return str(path)


def render_to_png(actors, path, size=(800, 600), scale=1,
                  bg_color=(0.15, 0.15, 0.15), camera=None):
    """Render ``actors`` offscreen and write a PNG to ``path``.

    ``actors`` is a single ``vtkActor`` (or ``TemporalActor``) or an iterable of
    them.  A renderer and an offscreen render window are created and thrown
    away, so this is a one-shot snapshot that needs no interactor and no
    ``start()`` loop.  ``camera`` may be a ``vtkCamera`` to copy the view from;
    otherwise the camera is reset to fit the scene.
    """
    if hasattr(actors, 'GetMapper') or hasattr(actors, 'actor'):
        actors = [actors]

    ren = vtk.vtkRenderer()
    ren.SetBackground(*bg_color)
    for actor in actors:
        ren.AddActor(actor.actor if hasattr(actor, 'actor') else actor)

    ren_win = vtk.vtkRenderWindow()
    ren_win.SetOffScreenRendering(1)
    ren_win.SetSize(int(size[0]), int(size[1]))
    ren_win.AddRenderer(ren)

    if camera is not None:
        ren.SetActiveCamera(camera)
    ren.ResetCamera()

    return _render_window_to_png(ren_win, path, scale=scale)
