"""Geometry checks for the LoD1 extrusion, and for ``examples/_lod1.py``.

Two independent oracles pin "is this shell a closed, outward-facing solid?":

* **signed volume** via the divergence theorem, V = 1/6 * sum(v0 . (v1 x v2)),
  over each face's triangles.  Positive means the normals point out of the
  solid; the value is the real volume when they do.
* **directed edges**: in a closed two-manifold shell every undirected edge
  appears exactly once in each direction.  An edge appearing twice one way is
  non-manifold (two lobes sharing a line), which is what a self-touching
  footprint produces.  vtkFeatureEdges does *not* catch this -- it still
  reports the shell watertight.

Both are checked against a 2 x 3 x 4 box, whose volume is exactly 24.
"""
import json

import numpy as np
import pytest

import _lod1
from _lod1 import (Building, ccw, clean, despike, height_of, lod1_scene,
                   n_boundary_edges, ring_area, solid_surfaces, write_cityjson)
from pyviz4d import read_cityjson

# 3 m across at Urbana (lon -88.23, lat 40.11) -- the case the old np.allclose
# relative tolerance destroyed.  3 m is ~2.7e-5 deg.
LON, LAT = -88.22806, 40.11493
D3M = 3.0 / _lod1.M_PER_DEG_LAT


# ------------------------------------------------------------------- oracles

def signed_volume(faces):
    """Volume of a closed shell given as faces (each an (n, 3) array)."""
    total = 0.0
    for face in faces:
        p = np.asarray(face, dtype=float)
        for k in range(1, len(p) - 1):
            total += np.dot(p[0], np.cross(p[k], p[k + 1])) / 6.0
    return total


def directed_edge_defects(index_faces):
    """Edges of a shell (as index rings) that break the manifold rule."""
    counts = {}
    for ring in index_faces:
        for j in range(len(ring)):
            e = (ring[j], ring[(j + 1) % len(ring)])
            counts[e] = counts.get(e, 0) + 1
    return [e for e, n in counts.items()
            if n != 1 or counts.get((e[1], e[0]), 0) != 1]


def box_shell(x, y, z):
    """(faces as index rings, faces as points) for an x by y by z box."""
    ring = ccw(np.array([[0.0, 0.0], [x, 0.0], [x, y], [0.0, y]]))
    floor, walls, roof = solid_surfaces(len(ring))
    points = ([(ring[i, 0], ring[i, 1], 0.0) for i in range(len(ring))]
              + [(ring[i, 0], ring[i, 1], z) for i in range(len(ring))])
    faces = [floor] + walls + [roof]
    return faces, [np.array([points[i] for i in f]) for f in faces]


# ------------------------------------------------------------ solid_surfaces

def test_box_volume_is_exactly_24():
    faces, point_faces = box_shell(2.0, 3.0, 4.0)
    assert signed_volume(point_faces) == pytest.approx(24.0)
    assert directed_edge_defects(faces) == []


@pytest.mark.parametrize("x,y,z", [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0),
                                   (0.5, 17.0, 2.25)])
def test_shells_are_closed_and_outward(x, y, z):
    faces, point_faces = box_shell(x, y, z)
    assert signed_volume(point_faces) == pytest.approx(x * y * z)
    assert directed_edge_defects(faces) == []


def test_inverted_ring_gives_negative_volume():
    """A clockwise ring makes the extrusion inside-out -- and the sign says so.

    This is the oracle that catches a ccw() regression: the faces are the same
    either way, only their winding changes.
    """
    ring = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    assert ring_area(ring) > 0                       # CCW as constructed
    floor, walls, roof = solid_surfaces(len(ring))
    points = ([(ring[i, 0], ring[i, 1], 0.0) for i in range(4)]
              + [(ring[i, 0], ring[i, 1], 4.0) for i in range(4)])
    good = [np.array([points[i] for i in f]) for f in [floor] + walls + [roof]]
    assert signed_volume(good) == pytest.approx(24.0)

    flipped = [f[::-1] for f in good]               # every normal reversed
    assert signed_volume(flipped) == pytest.approx(-24.0)


def test_lod1_scene_is_watertight():
    ring = ccw(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]))
    pd, actor, spans = lod1_scene([ring], [4.0], [(255, 0, 0, 255)])
    assert n_boundary_edges(pd) == 0
    assert pd.GetNumberOfPolys() == 6                # 4 walls + roof + floor
    assert spans == [(0, 6)]


# ------------------------------------------------------------------ ring hygiene

def test_clean_keeps_a_small_footprint():
    """Regression: np.allclose's default rtol ate whole small buildings.

    At a longitude of -88 deg the relative tolerance is ~78 m, so the first and
    last vertices of any footprint smaller than that compared "close" and one
    was deleted -- a triangle dropped below 3 vertices and vanished.  1109 of
    1801 UIUC buildings were lost that way before it was caught.
    """
    square = np.array([[LON, LAT], [LON + D3M, LAT],
                       [LON + D3M, LAT + D3M], [LON, LAT + D3M], [LON, LAT]])
    assert np.allclose(square[0], square[-1])        # the closing repeat
    # ...and, fatally, the old test also fired on two genuinely distinct points
    assert np.allclose(square[0], square[-2])
    assert len(clean(square)) == 4

    triangle = np.array([[LON, LAT], [LON + D3M, LAT], [LON + D3M, LAT + D3M]])
    assert len(clean(triangle)) == 3                 # must not fall below 3


def test_clean_drops_the_closing_repeat_and_dupes():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    assert len(clean(p)) == 3


def test_ccw_normalises_winding():
    cw = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
    assert ring_area(cw) < 0
    assert ring_area(ccw(cw)) > 0
    assert np.allclose(ccw(cw), cw[::-1])


def test_despike_cuts_the_degenerate_lobe():
    """A ring that runs out to a point and back is two lobes at one vertex.

    Kept, it puts two opposite-facing wall quads on one edge -- an edge with
    four faces -- and the pair z-fights in the renderer.
    """
    spike = np.array([[0.0, 0.0], [1.0, 0.0], [1.0 + 1e-7, 0.0], [1.0, 0.0],
                      [1.0, 1.0], [0.0, 1.0]])
    out = despike(spike)
    assert len(out) == 4
    assert abs(ring_area(out)) == pytest.approx(1.0, abs=1e-3)


def test_despike_leaves_a_clean_ring_alone():
    ring = ccw(np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]]))
    assert len(despike(ring)) == 4


# ------------------------------------------------------------------- height_of

DEFAULTS = {"yes": 3, "house": 2}


def test_height_of_prefers_measured():
    assert height_of({"height": "12 m"}, DEFAULTS) == (12.0, "height")
    assert height_of({"height": "12m"}, DEFAULTS) == (12.0, "height")


def test_height_of_falls_back_to_levels_then_the_table():
    assert height_of({"building:levels": "4"}, DEFAULTS) == (12.0, "building:levels")
    assert height_of({"building": "house"}, DEFAULTS) == (6.0, "default:house")
    assert height_of({}, DEFAULTS) == (9.0, "default:yes")


def test_height_of_drops_typos_instead_of_clamping():
    """A 4000 m 'height' is a typo; clamping it invents a plausible wrong answer."""
    assert height_of({"height": "4000"}, DEFAULTS) == (9.0, "default:yes")
    assert height_of({"height": "not a number"}, DEFAULTS) == (9.0, "default:yes")
    assert height_of({"building:levels": "200"}, DEFAULTS) == (9.0, "default:yes")


# ------------------------------------------------- write_cityjson / read back

def test_cityjson_round_trip_preserves_the_solid(tmp_path):
    """Write a box and a second box, read back, and re-run both oracles.

    Checked on the *file*: the nesting, the integer vertices under their
    transform, and that the reader agrees.
    """
    boxes = [("box_a", 2.0, 3.0, 4.0), ("box_b", 5.0, 5.0, 10.0)]
    buildings = []
    for key, x, y, z in boxes:
        ring = ccw(np.array([[0.0, 0.0], [x, 0.0], [x, y], [0.0, y]]))
        buildings.append(Building(key, ring, z, "height", {"name": key}))
    path = tmp_path / "boxes.city.json"
    write_cityjson(path, "two boxes", buildings, 0.0, 0.0)

    # read_cityjson accepts it (a file one nesting level short raises TypeError).
    # 15 points, not 16: both boxes sit on the origin, so vid() shares that
    # corner -- de-duplication working.
    pd = read_cityjson(str(path))
    assert pd.GetNumberOfPoints() == 15
    assert pd.GetNumberOfPolys() == 12

    data = json.loads(path.read_text())
    scale = data["transform"]["scale"]
    translate = data["transform"]["translate"]
    verts = np.asarray(data["vertices"], dtype=float) * scale + translate

    for key, x, y, z in boxes:
        solid = data["CityObjects"][key]["geometry"][0]
        shell = solid["boundaries"][0]
        # spec: boundaries -> shell -> surface -> ring -> [vertex index, ...]
        assert len(shell) == 6
        index_rings = []
        for surface in shell:
            assert len(surface) == 1
            index_rings.append(surface[0])

        assert directed_edge_defects(index_rings) == []
        assert signed_volume([verts[r] for r in index_rings]) == pytest.approx(x * y * z)
