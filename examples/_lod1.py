# Building data (c) OpenStreetMap contributors and/or Open City Model --
# see each demo_lod1_*.py for its own attribution and licence.
"""Shared LoD1 plumbing for the ``demo_lod1_*`` generators.

    uv run --extra geo python examples/demo_lod1_pudong.py      (etc.)

Four generators, one extrusion.  What differs per city is the *source*
(Overpass XML, Nominatim, Open City Model's S3 shards), the bbox and the
height table; everything from "footprint ring" to "closed shell, CityJSON
file, coloured actor" used to be copy-pasted four times, and had drifted --
one generator wound its rings unsigned, one wrote CityJSON one nesting level
short of spec (so pyviz4d's own read_cityjson() could not read it), and one
de-duplicated vertices with a *relative* tolerance that deleted whole small
buildings.  All of that lives here once.

Ring convention: ``(n, 2)`` lon/lat degrees, **counter-clockwise**, with the
closing repeat of the first vertex already removed.  ``clean()`` then
``ccw()`` get you there from a raw OSM/OCM ring.  The extrusion is
outward-facing only for a CCW ring.

This module is imported as ``import _lod1``: the generators are run as
scripts, so Python puts ``examples/`` on ``sys.path``.
"""
import json
import math
from collections import namedtuple
from pathlib import Path

import numpy as np
import vtk

from pyviz4d import viewpoint  # noqa: F401  (re-exported for the generators)

M_PER_DEG_LAT = 110574.0
LEVEL_M = 3.0

# A lobe this small (m^2) is coordinate rounding, not a wing.
MIN_LOBE_M2 = 0.1

# Vertex de-duplication tolerance, 1e-9 deg ~ 0.1 mm.  ABSOLUTE, and this
# matters: np.allclose's default rtol is 1e-5, which at a longitude of 121 deg
# is ~110 m and at -88 deg is ~78 m.  Used on a lon/lat ring it will call the
# first and last vertices of any small footprint "close enough" and delete a
# real one -- a three-gon then drops below the minimum and the building
# vanishes.  (This cost the UIUC ingest 1109 of 1801 buildings before it was
# caught.)
DUP = dict(rtol=0.0, atol=1e-9)

# One building: key is a stable id for the CityJSON object name, ring is CCW
# lon/lat as above, source says where the height came from, attrs are extra
# CityJSON attributes beyond height_m / height_source.
Building = namedtuple("Building", "key ring height source attrs")


def clean(p):
    """Drop repeated vertices from a ring.

    Sources round footprints to ~1e-6 deg (0.1 m) and close a ring by
    repeating its first vertex, so neighbours can coincide exactly.  Compare
    with :data:`DUP` -- an absolute tolerance -- never with np.allclose's
    default relative one (see there).
    """
    p = np.asarray(p, float)
    if len(p) > 1 and np.allclose(p[0], p[-1], **DUP):
        p = p[:-1]
    keep = np.r_[True, (np.abs(np.diff(p, axis=0)).sum(1) > DUP["atol"])]
    p = p[keep]
    if len(p) > 1 and np.allclose(p[0], p[-1], **DUP):
        p = p[:-1]
    return p


def ring_area(p):
    """Signed area of a ring in its own units (positive = counter-clockwise)."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ccw(p):
    """Wring the ring counter-clockwise.

    Sources do not wind footprints consistently -- OCM is ~4 per cent
    clockwise, OSM arbitrary -- and the floor/wall/roof construction in
    :func:`solid_surfaces` is only outward-facing for a CCW ring.
    """
    return p[::-1] if ring_area(p) < 0 else p


def despike(p, min_lobe=MIN_LOBE_M2):
    """Cut zero-area out-and-back excursions out of a footprint ring.

    Coordinate rounding can send a ring out to a vertex and straight back.  A
    ring visited twice at one point splits into two lobes there; cut whichever
    encloses (almost) no area and keep the other.  Left in, the two lobes put
    two opposite-facing wall quads on one edge -- a non-manifold shell that
    vtkFeatureEdges still calls watertight.  A no-op on clean rings.
    """
    p = clean(p)
    while len(p) >= 3:
        nxt = None
        for i in range(len(p)):
            for j in range(i + 2, len(p)):
                if np.hypot(*(p[i] - p[j])) > DUP["atol"]:
                    continue
                fwd, rev = p[i:j + 1], np.r_[p[j:], p[:i + 1]]
                if abs(ring_area(fwd)) <= min_lobe:
                    nxt = rev
                elif abs(ring_area(rev)) <= min_lobe:
                    nxt = fwd
                if nxt is not None:
                    break
            if nxt is not None:
                break
        if nxt is None:
            return p
        p = clean(nxt)
    return p


def height_of(tags, defaults, height_range=(2.0, 700.0),
              levels_range=(0.5, 200.0), level_m=LEVEL_M):
    """Building height in metres from OSM tags.

    ``height`` if it parses and lands inside ``height_range``, else
    ``building:levels`` x ``level_m`` inside ``levels_range``, else
    ``defaults[building_type]`` (a per-city table -- a Shanghai tower block
    and a Hainan campus do not share floor counts) and finally
    ``defaults['yes']``.  Returns ``(height_m, source)`` where source is the
    tag key, or ``'default:<type>'`` when the defaults table decided.

    Out-of-range readings are dropped rather than clamped: a 4000 m "height"
    or a 200-storey "levels" is a typo, and clamping it to the limit silently
    invents a plausible-looking wrong answer.
    """
    for key, scale, lo, hi in (("height", 1.0, *height_range),
                               ("building:levels", level_m, *levels_range)):
        v = tags.get(key)
        if not v:
            continue
        try:
            f = float(str(v).split(";")[0].replace("m", "").strip().split()[0]) * scale
        except ValueError:
            continue
        if lo <= f <= hi:
            return f, key
    bt = tags.get("building", "yes")
    return defaults.get(bt, defaults["yes"]) * level_m, "default:" + bt


def _surface(ids, idx):
    """One CityJSON surface -- ``[ring]`` -- with the ring mapped through ``ids``."""
    return [[ids[i] for i in idx]]


def solid_surfaces(n):
    """The closed LoD1 shell of an n-gon extrusion, as index lists.

    Indices run over ``[bot..., top...]`` with ``bot = 0..n-1`` and
    ``top = n..2n-1``.  Returns ``(floor, walls, roof)``: ``floor`` and
    ``roof`` are rings, ``walls`` a list of quads.  All are wound
    counter-clockwise seen from outside -- floor first so its normal points
    down, roof last so its normal points up -- but only if the ring is CCW
    (see :func:`ccw`).
    """
    walls = [[i, (i + 1) % n, n + (i + 1) % n, n + i] for i in range(n)]
    return list(reversed(range(n))), walls, list(range(n, 2 * n))


def write_cityjson(path, title, buildings, lon0, lat0):
    """Write ``buildings`` as CityJSON 1.1 and return (solids, verts, bytes).

    Integer vertices under a ``transform``, so the file stays small and exact.

    Nesting follows the spec: a Solid's ``boundaries`` is a list of shells, a
    shell a list of surfaces, a surface a list of rings -- i.e.
    ``[[[ring], ...], ...]``.  Writing ``[surface, ...]`` instead (one level
    short) produces a file that pyviz4d's own read_cityjson() rejects with
    ``TypeError: object of type 'int' has no len()``.
    """
    scale = [1e-7, 1e-7, 0.001]
    translate = [round(lon0, 5), round(lat0, 5), 0.0]
    verts, vmap, cos = [], {}, {}

    def vid(lon, lat, z):
        key = (round((lon - translate[0]) / scale[0]),
               round((lat - translate[1]) / scale[1]), round(z / scale[2]))
        if key not in vmap:
            vmap[key] = len(verts)
            verts.append([int(key[0]), int(key[1]), int(key[2])])
        return vmap[key]

    for b in buildings:
        n = len(b.ring)
        bot = [vid(b.ring[i, 0], b.ring[i, 1], 0.0) for i in range(n)]
        top = [vid(b.ring[i, 0], b.ring[i, 1], b.height) for i in range(n)]
        floor, walls, roof = solid_surfaces(n)
        ids = bot + top
        shell = ([_surface(ids, floor)]               # a shell = [surface, ...]
                 + [_surface(ids, w) for w in walls]
                 + [_surface(ids, roof)])
        attrs = {"height_m": round(b.height, 2), "height_source": b.source}
        attrs.update(b.attrs)
        cos[b.key] = {"type": "Building", "attributes": attrs,
                      "geometry": [{"type": "Solid", "lod": "1",
                                    "boundaries": [shell]}]}

    cj = {"type": "CityJSON", "version": "1.1",
          "transform": {"scale": scale, "translate": translate},
          "metadata": {"referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4326",
                       "title": title},
          "CityObjects": cos, "vertices": verts}
    path = Path(path)
    path.write_text(json.dumps(cj, separators=(",", ":")))
    print(f"CityJSON: {len(cos)} solids, {len(verts)} vertices, "
          f"{path.stat().st_size/1e6:.2f} MB")
    return len(cos), len(verts), path.stat().st_size


def lod1_scene(rings_enu, heights, rgbs, edge_rgb=(0.03, 0.03, 0.05),
               edge_width=0.4):
    """Extrude ENU rings into closed shells and colour them by building.

    ``rings_enu`` are CCW rings in local metres (see :func:`enu`), ``heights``
    and ``rgbs`` parallel lists -- ``rgbs[i]`` an ``(r, g, b, a)`` tuple of
    0-255 ints, shared by every cell of building i.  Returns
    ``(polydata, actor, spans)``, where ``spans[i]`` is building i's cell range
    (its walls, then roof, then floor) so a caller can recolour one building in
    place -- see demo_lod1_uiuc.py's ``h`` toggle, which paints the landmark
    into this array rather than drawing a second copy of its faces over them
    (two copies of one surface z-fight, and the landmark flashes).
    """
    pts, cells, cols = vtk.vtkPoints(), vtk.vtkCellArray(), []
    spans = []
    for ring, h, rgb in zip(rings_enu, heights, rgbs):
        c0 = cells.GetNumberOfCells()
        n = len(ring)
        bot = [pts.InsertNextPoint(ring[i, 0], ring[i, 1], 0.0) for i in range(n)]
        top = [pts.InsertNextPoint(ring[i, 0], ring[i, 1], h) for i in range(n)]
        floor, walls, roof = solid_surfaces(n)
        ids = bot + top
        for idx in walls + [roof, floor]:            # walls, then roof, then floor
            idl = vtk.vtkIdList()
            for k in idx:
                idl.InsertNextId(ids[k])
            cells.InsertNextCell(idl)
            cols.append(rgb)
        spans.append((c0, cells.GetNumberOfCells()))

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    rgba = vtk.vtkUnsignedCharArray()
    rgba.SetNumberOfComponents(4)
    rgba.SetName("colors")
    for c in cols:
        rgba.InsertNextTuple4(*c)
    pd.GetCellData().SetScalars(rgba)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarRange(0, 255)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetEdgeVisibility(1)                 # dark edges read the LoD1 boxes
    prop.SetEdgeColor(*edge_rgb)
    prop.SetLineWidth(edge_width)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.85)
    return pd, actor, spans


def n_boundary_edges(pd):
    """Count the open edges of ``pd``: 0 means the shells are closed."""
    fe = vtk.vtkFeatureEdges()
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    return fe.GetOutput().GetNumberOfCells()


def enu(ring, lon0, lat0):
    """(n, 2) lon/lat ring -> local east/north metres about (lon0, lat0)."""
    coslat = math.cos(math.radians(lat0))
    return np.c_[(ring[:, 0] - lon0) * M_PER_DEG_LAT * coslat,
                 (ring[:, 1] - lat0) * M_PER_DEG_LAT]


def oblique_camera(cx, cy, span, z_focus=0.30, dx=0.62, dy=-0.95, dz=0.58):
    """The low, oblique eye these models want.

    A flat campus or a skyline both hide their extrusion under a straight-down
    ResetCamera, so aim from the usual 3/4 view.  Note that only the camera's
    direction and view-up survive into a PNG -- render_to_png() refits the
    distance -- so ``span`` sets the eye *elevation*, not the framing.  Use
    :func:`pyviz4d.viewpoint` to hand the same view to a Viewer4D.
    """
    cam = vtk.vtkCamera()
    cam.SetFocalPoint(cx, cy, z_focus * span)
    cam.SetPosition(cx + dx * span, cy + dy * span, dz * span)
    cam.SetViewUp(0, 0, 1)
    return cam


def png_stats(path, size, pd):
    """Print the standard "did that render?" line for a written PNG."""
    import imageio.v2 as iio
    img = iio.imread(path).reshape(-1, 3)
    uniq = len(np.unique(img, axis=0))
    bg = np.abs(img.astype(int) - [38, 38, 38]).sum(1) <= 12
    print(f"PNG {path}: {size[0]}x{size[1]}, {pd.GetNumberOfPoints()} points, "
          f"{pd.GetNumberOfPolys()} faces, unique colours {uniq}, "
          f"non-background {1 - bg.mean():.3f}")
