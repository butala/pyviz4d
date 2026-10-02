# Building data (c) Open City Model, BuildZero - ODbL 1.0
#   https://registry.opendata.aws/opencitymodel/  (footprints: Microsoft USBuildingFootprints)
#!/usr/bin/env python3
"""UIUC campus (Urbana-Champaign, Illinois) in pyviz4d, from Open City Model.

    uv run --extra geo python examples/demo_lod1_uiuc.py [--interactive]

Open City Model publishes LoD1 CityJSON for every US county on S3 (no AWS
account needed); Champaign County is Illinois/17019, ~68 MB across two files.
Each Building carries an extruded Solid, so unlike the OSM generators this one
needs no footprints or heights of its own -- it reads them, keeps the campus
bbox, and rebuilds the shells so the winding is outward-facing throughout.
WGS84 (EPSG:4979) throughout; local ENU metres for the geometry.

Reading the solids as they stand is not quite enough.  OCM's winding is not
uniform: almost all of these solids are already outward, but a few per cent
arrive inside-out, so rather than trust the source the script re-winds each
footprint ring outward (see ccw()).  A handful of rings also touch themselves
-- the footprint simplification runs out to a vertex and straight back,
enclosing a sliver a few cm^2 in area -- which leaves the closed shell
non-manifold; those lobes are cut out (see despike()).

Caveat worth reading before trusting the vertical: OCM's heights are modelled,
not measured (`height_source` is `msfp-2017` or `model`), and the tail is wild
-- elsewhere in this same county file a single solid stands 261 m tall.  The
campus bbox peaks at 88 m against a 99th percentile of 35 m, and the top four
solids (61-88 m) are a suspicious jump; --max-height trims such candidates if
you want them gone (at its 120 m default it fires nowhere in this bbox).
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import requests
import vtk
from matplotlib import colormaps
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parents[1] / "data" / "uiuc"
HERE.mkdir(parents=True, exist_ok=True)
BUCKET = "https://opencitymodel.s3.amazonaws.com"
PREFIX = "2019-jun/json/Illinois/17019"          # Champaign County, FIPS 17019
SHARDS = ("Illinois-17019-000.json", "Illinois-17019-001.json")
# Main Quad (40.1095, -88.2272) with the engineering campus west and Campustown east
BBOX = (-88.2400, 40.1000, -88.2160, 40.1200)    # lon0, lat0, lon1, lat1
UA = {"User-Agent": "pyviz4d-uiuc/0.1 (research)"}
M_PER_DEG_LAT = 110574.0
MIN_HEIGHT = 2.0

# Landmarks that --highlight can pick out, as (lat, lon) of the building's
# centre.  OSM/Nominatim has the ECE Building (306 N Wright St, 230,000 sq ft
# over five floors) at 40.11493, -88.22806.
LANDMARKS = {"ece": (40.11493, -88.22806)}


def fetch(name):
    """Download one county shard once; the county is ~68 MB, so always cache."""
    cache = HERE / name
    if not cache.exists():
        r = requests.get(f"{BUCKET}/{PREFIX}/{name}", headers=UA, timeout=600)
        r.raise_for_status()
        cache.write_bytes(r.content)
        print(f"downloaded {name} ({cache.stat().st_size/1e6:.1f} MB)")
    return json.loads(cache.read_text())


def clean(p):
    """Drop repeated vertices; OCM rounds footprints to ~1e-6 deg (0.1 m), so
    neighbours in a ring can coincide exactly.

    Compare with an ABSOLUTE tolerance.  ``np.allclose``'s default relative
    tolerance is 1e-5, which on a longitude of -88.23 deg is ~78 m -- big enough
    to delete a small building's whole outline and silently drop it (this cost
    the UIUC ingest 1109 of 1801 buildings before it was caught).
    """
    dup = dict(rtol=0.0, atol=1e-9)            # 1e-9 deg ~ 0.1 mm
    if len(p) > 1 and np.allclose(p[0], p[-1], **dup):
        p = p[:-1]
    keep = np.r_[True, (np.abs(np.diff(p, axis=0)).sum(1) > 1e-9)]
    p = p[keep]
    if len(p) > 1 and np.allclose(p[0], p[-1], **dup):
        p = p[:-1]
    return p


def ring_area(p):
    """Signed area of a ring (positive = counter-clockwise)."""
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


# A lobe this small (m^2) is OCM's 1e-6 deg rounding, not a wing.  Champaign
# County has four of them, in two buildings, at 0.000 and 0.025 m^2.
MIN_LOBE_M2 = 0.1


def despike(p, min_lobe=MIN_LOBE_M2):
    """Cut the zero-area out-and-back excursions out of a footprint ring.

    OCM's footprints are snapped to a 1e-6 deg grid, and the snapping sometimes
    makes a ring touch itself: it walks out to a vertex and comes straight back
    (sometimes with a vertex or two on the return leg), leaving a sliver of a
    few cm^2.  Reproduced literally such a ring gives a non-manifold shell --
    two opposite-facing wall quads share an edge with the real wall, so that
    edge borders four faces, and the coincident quads z-fight in the renderer.
    vtk still counts the shell as watertight; a directed-edge check does not.

    A ring visited twice at one point splits into two lobes; cut whichever
    encloses (almost) no area and keep the other.  A genuine figure-of-eight
    footprint has lobes of hundreds of m^2 and is left alone.
    """
    p = clean(np.asarray(p, float))
    while len(p) >= 3:
        nxt = None
        for i in range(len(p)):
            for j in range(i + 2, len(p)):
                if np.hypot(*(p[i] - p[j])) > 1e-9:
                    continue                  # p[i], p[j] are different points
                fwd, rev = p[i:j + 1], np.r_[p[j:], p[:i + 1]]
                if abs(ring_area(fwd)) <= min_lobe:
                    nxt = rev                 # drop the out-and-back
                elif abs(ring_area(rev)) <= min_lobe:
                    nxt = fwd                 # drop its complement
                if nxt is not None:
                    break
            if nxt is not None:
                break
        if nxt is None:
            return p                          # no degenerate lobe left
        p = clean(nxt)                        # both candidates are strictly shorter
    return p


def ccw(p):
    """OCM does not wind its footprint rings consistently -- most are already
    counter-clockwise, a few per cent are clockwise -- so normalise: the
    floor/wall/roof construction below is only outward-facing for a
    counter-clockwise ring."""
    return p[::-1] if ring_area(p) < 0 else p


def viewpoint(ren, cam):
    """Give a Viewer4D the same view the offscreen render uses.

    Viewer4D never fits its camera: the renderer starts at VTK's default
    position (0, 0, 1) looking down -z, and start() latches that as the initial
    view (so the `r` hotkey restores the empty one).  render_to_png() ends with
    ren.ResetCamera(), which keeps only a camera's *direction* and view-up, so
    copy those across and let the renderer refit the distance.
    """
    vcam = ren.GetActiveCamera()
    vcam.SetPosition(*cam.GetPosition())
    vcam.SetFocalPoint(*cam.GetFocalPoint())
    vcam.SetViewUp(*cam.GetViewUp())
    ren.ResetCamera()
    ren.ResetCameraClippingRange()


def ingest(max_height):
    """Campus buildings as (footprint lon/lat ring, height, source, attributes)."""
    buildings, dropped, srcs, total = [], 0, {}, 0
    for name in SHARDS:
        data = fetch(name)
        verts = np.asarray(data["vertices"], dtype=np.float64)
        for oid, obj in data["CityObjects"].items():
            a = obj["attributes"]
            lon, lat = a.get("longitude"), a.get("latitude")
            if lon is None or lat is None:
                continue
            if not (BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]):
                continue
            total += 1
            h = float(a.get("height") or a.get("measuredHeight") or 0.0)
            if h < MIN_HEIGHT or (max_height and h > max_height):
                dropped += 1
                continue
            geom = (obj.get("geometry") or [{}])[0]
            if geom.get("type") != "Solid":
                dropped += 1
                continue
            # shell is [floor, wall surfaces..., roof]; the roof ring is the outline
            roof = geom["boundaries"][0][-1][0]
            p = despike(verts[roof][:, :2])
            if len(p) < 3:
                dropped += 1
                continue
            src = a.get("height_source", "?")
            srcs[src] = srcs.get(src, 0) + 1
            buildings.append((oid, ccw(p), h, src, a))
    return buildings, total, dropped, srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=str(HERE / "uiuc.png"))
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--max-height", type=float, default=120.0,
                    help="drop OCM solids taller than this (0 keeps everything); "
                         "the modelled heights have a long tail of mis-models")
    ap.add_argument("--interactive", action="store_true",
                    help="open a Viewer4D window instead of writing a PNG")
    ap.add_argument("--highlight", default="ece", choices=sorted(LANDMARKS),
                    help="landmark to paint bright (default ece); `h` in the "
                         "interactive window toggles it")
    ap.add_argument("--no-highlight", action="store_true",
                    help="start with no landmark highlighted")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.lower().split("x"))

    buildings, total, dropped, srcs = ingest(args.max_height)
    lat0 = (BBOX[1] + BBOX[3]) / 2
    lon0 = (BBOX[0] + BBOX[2]) / 2
    coslat = math.cos(math.radians(lat0))

    buildings.sort(key=lambda b: -b[2])
    heights = np.array([b[2] for b in buildings])
    print(f"Open City Model {PREFIX}: {total} buildings in bbox, "
          f"{dropped} skipped, {len(buildings)} kept")
    print(f"height source {srcs}")
    print(f"height m: min {heights.min():.1f} med {np.median(heights):.1f} "
          f"max {heights.max():.1f}")
    print("tallest:")
    for oid, p, h, src, a in buildings[:10]:
        print(f"  {h:6.1f} m  {len(p):3d}-gon {a.get('area', 0):7.0f} m2  "
              f"ubid {a.get('ubid')}  [{src}]")

    def enu(p):
        return np.c_[(p[:, 0] - lon0) * M_PER_DEG_LAT * coslat,
                     (p[:, 1] - lat0) * M_PER_DEG_LAT]

    # ---- CityJSON ----
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

    for oid, p, h, src, a in buildings:
        n = len(p)
        bot = [vid(p[i, 0], p[i, 1], 0.0) for i in range(n)]
        top = [vid(p[i, 0], p[i, 1], h) for i in range(n)]
        # CityJSON nests a Solid as boundaries = [shell], shell = [surface],
        # surface = [ring], so each surface here is a one-element list.  (The
        # three OSM generators write boundaries = [surface, ...] instead, one
        # level short; demo_lod1_view.py tolerates both, read_cityjson does not.)
        surfaces = ([[bot[::-1]]]                               # floor, normal down
                    + [[[bot[i], bot[(i + 1) % n],
                         top[(i + 1) % n], top[i]]] for i in range(n)]  # walls, out
                    + [[top]])                                  # roof, normal up
        cos["ocm" + oid] = {"type": "Building",
                            "attributes": {"height_m": round(h, 2),
                                           "height_source": src,
                                           "footprint_source": a.get("fp_source"),
                                           "footprint_area_m2": round(a.get("area", 0.0), 1),
                                           "ubid": a.get("ubid"),
                                           "ocm_id": oid},
                            "geometry": [{"type": "Solid", "lod": "1",
                                          "boundaries": [surfaces]}]}
    cj = {"type": "CityJSON", "version": "1.1",
          "transform": {"scale": scale, "translate": translate},
          "metadata": {"referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4326",
                       "title": "UIUC campus, Urbana-Champaign IL - LoD1 from Open City Model"},
          "CityObjects": cos, "vertices": verts}
    cj_path = HERE / "uiuc_lod1.city.json"
    cj_path.write_text(json.dumps(cj, separators=(",", ":")))
    print(f"CityJSON: {len(cos)} solids, {len(verts)} vertices, "
          f"{cj_path.stat().st_size/1e6:.2f} MB")

    # ---- render ----
    norm = LogNorm(max(heights.min(), 3.0), heights.max())
    cmap = colormaps["turbo"]
    pts, cells, cols = vtk.vtkPoints(), vtk.vtkCellArray(), []
    for oid, p, h, src, a in buildings:
        e = enu(p)                                # p is CCW, so all normals point out
        n = len(e)
        bot = [pts.InsertNextPoint(e[i, 0], e[i, 1], 0.0) for i in range(n)]
        top = [pts.InsertNextPoint(e[i, 0], e[i, 1], h) for i in range(n)]
        r, g, b, _ = cmap(norm(h))
        rgb = (int(r * 255), int(g * 255), int(b * 255), 255)
        for i in range(n):                        # walls
            j = (i + 1) % n
            ids = vtk.vtkIdList()
            for k in (bot[i], bot[j], top[j], top[i]):
                ids.InsertNextId(k)
            cells.InsertNextCell(ids)
            cols.append(rgb)
        ids = vtk.vtkIdList()                     # roof
        for k in top:
            ids.InsertNextId(k)
        cells.InsertNextCell(ids)
        cols.append(rgb)
        ids = vtk.vtkIdList()                     # floor: closes the shell
        for k in reversed(bot):
            ids.InsertNextId(k)
        cells.InsertNextCell(ids)
        cols.append(rgb)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    rgba = vtk.vtkUnsignedCharArray()
    rgba.SetNumberOfComponents(4)
    rgba.SetName("colors")
    for c in cols:
        rgba.InsertNextTuple4(*c)
    pd.GetCellData().SetScalars(rgba)

    fe = vtk.vtkFeatureEdges()                    # 0 boundary edges = closed shells
    fe.SetInputData(pd)
    fe.BoundaryEdgesOn()
    fe.FeatureEdgesOff()
    fe.NonManifoldEdgesOff()
    fe.ManifoldEdgesOff()
    fe.Update()
    n_open = fe.GetOutput().GetNumberOfCells()
    print(f"shells: {len(buildings)} buildings, {n_open} boundary edges "
          f"({'watertight' if n_open == 0 else 'NOT watertight'})")

    # ---- landmark highlight: a separate actor, so `h` can toggle it ----
    hi = None
    if not args.no_highlight:
        lat, lon = LANDMARKS[args.highlight]
        want = np.array([(lon - lon0) * M_PER_DEG_LAT * coslat,
                         (lat - lat0) * M_PER_DEG_LAT])
        near = [np.hypot(*(enu(b[1]).mean(0) - want)) for b in buildings]
        k = int(np.argmin(near))
        oid, p, h, src, a = buildings[k]
        print(f"highlight {args.highlight}: ubid {a.get('ubid')} "
              f"({h:.1f} m, {a.get('area', 0):.0f} m2), {near[k]:.0f} m from "
              f"the landmark")

        e = enu(p)
        n = len(e)
        hpts, hpolys, hlines = vtk.vtkPoints(), vtk.vtkCellArray(), vtk.vtkCellArray()
        bot = [hpts.InsertNextPoint(e[i, 0], e[i, 1], 0.0) for i in range(n)]
        top = [hpts.InsertNextPoint(e[i, 0], e[i, 1], h) for i in range(n)]
        for i in range(n):                       # walls and roof, floor omitted
            j = (i + 1) % n
            ids = vtk.vtkIdList()
            for v in (bot[i], bot[j], top[j], top[i]):
                ids.InsertNextId(v)
            hpolys.InsertNextCell(ids)
        ids = vtk.vtkIdList()
        for v in top:
            ids.InsertNextId(v)
        hpolys.InsertNextCell(ids)

        cx, cy = e.mean(0)                       # a pole, so the pick reads
        pole = vtk.vtkIdList()                   # from any angle and any zoom
        pole.InsertNextId(hpts.InsertNextPoint(cx, cy, h))
        pole.InsertNextId(hpts.InsertNextPoint(cx, cy, h + 0.35 * heights.max()))
        hlines.InsertNextCell(pole)

        hpd = vtk.vtkPolyData()
        hpd.SetPoints(hpts)
        hpd.SetPolys(hpolys)
        hpd.SetLines(hlines)
        hm = vtk.vtkPolyDataMapper()
        hm.SetInputData(hpd)
        hm.SetResolveCoincidentTopologyToPolygonOffset()   # no z-fight on the faces
        hi = vtk.vtkActor()
        hi.SetMapper(hm)
        hp = hi.GetProperty()
        hp.SetColor(1.0, 0.0, 0.6)               # magenta: absent from turbo
        hp.SetEdgeVisibility(1)
        hp.SetEdgeColor(1.0, 1.0, 1.0)
        hp.SetLineWidth(1.5)
        hp.SetAmbient(0.7)
        hp.SetDiffuse(0.6)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()
    mapper.SetScalarRange(0, 255)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetEdgeVisibility(1)
    prop.SetEdgeColor(0.03, 0.03, 0.05)
    prop.SetLineWidth(0.4)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.85)

    # a flat campus needs a low, oblique eye: ~55 deg azimuth, ~25 deg elevation
    span = max(np.ptp(enu(np.array([[BBOX[0], BBOX[1]],
                                    [BBOX[2], BBOX[3]]])), axis=0).max(),
               heights.max())
    cx, cy = 0.0, 0.0                         # the bbox centre is the ENU origin
    cam = vtk.vtkCamera()
    cam.SetFocalPoint(cx, cy, 0.04 * span)
    cam.SetPosition(cx + 0.62 * span, cy - 0.88 * span, 0.34 * span)
    cam.SetViewUp(0, 0, 1)
    # NOTE: only the camera's *direction* and view-up survive --
    # render_to_png() ends with ren.ResetCamera(), which recomputes the distance
    # from the current view angle and therefore cancels any SetPosition radius
    # or pre-applied Zoom().  For a tighter frame, use the interactive viewer.

    from pyviz4d import render_to_png
    if args.interactive:
        from pyviz4d import Viewer4D
        viewer = Viewer4D(size=(W, H), bg_color=(0.12, 0.12, 0.14))
        viewer.add_actor(actor)
        if hi is not None:
            viewer.add_actor(hi)

            def on_key(obj, _event):
                if obj.GetKeySym().lower() == "h":
                    hi.SetVisibility(0 if hi.GetVisibility() else 1)
                    obj.GetRenderWindow().Render()

            # Viewer4D binds f and r itself at priority 1.0; `h` is free
            viewer.iren.AddObserver("KeyPressEvent", on_key)
            print(f"`h` toggles the {args.highlight} highlight")
        viewpoint(viewer.ren, cam)   # else the window opens on the default camera
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
        viewer.start()
        return 0

    render_to_png([actor] + ([hi] if hi is not None else []),
                  args.png, size=(W, H), camera=cam)

    import imageio.v2 as iio
    img = iio.imread(args.png).reshape(-1, 3)
    uniq = len(np.unique(img, axis=0))
    bg = np.abs(img.astype(int) - [38, 38, 38]).sum(1) <= 12
    print(f"PNG {args.png}: {W}x{H}, {pd.GetNumberOfPoints()} points, "
          f"{pd.GetNumberOfPolys()} faces, unique colours {uniq}, "
          f"non-background {1 - bg.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
