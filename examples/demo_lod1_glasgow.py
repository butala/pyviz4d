# Building data (c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright
#!/usr/bin/env python3
"""James Watt Building area, University of Glasgow - LoD1 from OSM.

    python data/glasgow/make_glasgow.py [--radius 350] [--png out.png]

Geocodes the place with Nominatim, pulls OSM within +/- radius metres, extrudes
every building footprint to a LoD1 solid, writes CityJSON, and renders a still
with pyviz4d.  Add --interactive to open a Viewer4D window instead.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parents[1] / "data" / "glasgow"
HERE.mkdir(parents=True, exist_ok=True)
PLACE = "James Watt Building, Glasgow"
FALLBACK = (55.87165, -4.29135)          # University Avenue, Gilmorehill
UA = {"User-Agent": "pyviz4d-glasgow/0.1 (research)"}
M_PER_DEG_LAT = 110574.0
LEVEL_M = 3.0
DEFAULT_LEVELS = {"yes": 3, "house": 2, "residential": 4, "apartments": 6,
                  "dormitory": 5, "school": 3, "university": 4, "college": 4,
                  "commercial": 4, "retail": 2, "hotel": 8, "hospital": 5,
                  "industrial": 2, "warehouse": 3, "construction": 3,
                  "office": 5, "civic": 4, "public": 4, "church": 8}


def geocode():
    try:
        r = requests.get("https://nominatim.openstreetmap.org/search",
                         params={"q": PLACE, "format": "json", "limit": 1},
                         headers=UA, timeout=30)
        hit = r.json()[0]
        b = [float(v) for v in hit["boundingbox"]]         # minlat, maxlat, ...
        return (b[0] + b[1]) / 2, (b[2] + b[3]) / 2, hit["display_name"]
    except Exception as exc:                                # offline / rate limit
        print("geocode failed (%s); using fallback coords" % exc)
        return FALLBACK[0], FALLBACK[1], "fallback: University Avenue, Gilmorehill"


def height_of(tags):
    for key, scale, lo, hi in (("height", 1.0, 2.0, 200.0),
                               ("building:levels", LEVEL_M, 0.5, 60.0)):
        v = tags.get(key)
        if not v:
            continue
        try:
            f = float(str(v).split(";")[0].replace("m", "").strip().split()[0]) * scale
            if lo <= f <= hi:
                return f, key
        except ValueError:
            pass
    bt = tags.get("building", "yes")
    return DEFAULT_LEVELS.get(bt, 3) * LEVEL_M, "default:" + bt


def clean(p):
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    p = p[np.r_[True, (np.abs(np.diff(p, axis=0)).sum(1) > 1e-9)]]
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=350.0, help="metres around the centre")
    ap.add_argument("--png", default=str(HERE / "glasgow.png"))
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.lower().split("x"))

    lat, lon, name = geocode()
    dlat = args.radius / M_PER_DEG_LAT
    dlon = args.radius / (M_PER_DEG_LAT * math.cos(math.radians(lat)))
    bbox = (lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    print(f"centre: {name}")
    print(f"lat {lat:.6f} lon {lon:.6f} | bbox {bbox[0]:.5f},{bbox[1]:.5f},{bbox[2]:.5f},{bbox[3]:.5f} "
          f"({2*args.radius:.0f} m across)")

    cache = HERE / "jameswatt.osm"
    if cache.exists():
        xml = cache.read_text()
    else:
        r = requests.get("https://api.openstreetmap.org/api/0.6/map",
                         params={"bbox": ",".join(f"{v:.6f}" for v in bbox)},
                         headers=UA, timeout=300)
        r.raise_for_status()
        cache.write_text(r.text)
        xml = r.text

    root = ET.fromstring(xml)
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat")))
             for n in root.findall("node")}
    buildings = []
    for w in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        if "building" not in tags:
            continue
        pts = [nodes[r] for r in (nd.get("ref") for nd in w.findall("nd")) if r in nodes]
        if len(pts) < 4:
            continue
        p = clean(np.asarray(pts))
        if len(p) < 3:
            continue
        h, src = height_of(tags)
        buildings.append((w.get("id"), p, h, src, tags))

    def enu(p):
        return np.c_[(p[:, 0] - lon) * M_PER_DEG_LAT * math.cos(math.radians(lat)),
                     (p[:, 1] - lat) * M_PER_DEG_LAT]

    buildings.sort(key=lambda b: -b[2])
    heights = np.array([b[2] for b in buildings])
    srcs = {}
    for b in buildings:
        srcs[b[3]] = srcs.get(b[3], 0) + 1
    print(f"buildings {len(buildings)} | height source {srcs}")
    print(f"height m: max {heights.max():.1f} med {np.median(heights):.1f}")
    print("tallest:")
    for wid, p, h, src, tags in buildings[:8]:
        nm = tags.get("name") or "(unnamed)"
        print(f"  {h:6.1f} m  {nm[:46]:46s} {tags.get('building')} [{src}]")

    # ---- CityJSON LoD1 ----
    scale = [1e-7, 1e-7, 0.001]
    tr = [round(lon, 6), round(lat, 6), 0.0]
    verts, vmap, cos_ = [], {}, {}

    def vid(la, lo, z):
        k = (round((lo - tr[0]) / scale[0]), round((la - tr[1]) / scale[1]), round(z / scale[2]))
        if k not in vmap:
            vmap[k] = len(verts)
            verts.append([int(k[0]), int(k[1]), int(k[2])])
        return vmap[k]

    for wid, p, h, src, tags in buildings:
        n = len(p)
        bot = [vid(p[i, 1], p[i, 0], 0.0) for i in range(n)]
        top = [vid(p[i, 1], p[i, 0], h) for i in range(n)]
        shell = [bot] + [[bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]]
                         for i in range(n)] + [top[::-1]]
        cos_["b" + wid] = {"type": "Building",
                           "attributes": {"height_m": round(h, 2), "height_source": src,
                                          "osm_way": wid, "name": tags.get("name"),
                                          "building": tags.get("building", "yes")},
                           "geometry": [{"type": "Solid", "lod": "1",
                                         "boundaries": [shell]}]}
    cj = {"type": "CityJSON", "version": "1.1",
          "transform": {"scale": scale, "translate": tr},
          "metadata": {"referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4326",
                       "title": "James Watt Building area, University of Glasgow - LoD1 (OSM)"},
          "CityObjects": cos_, "vertices": verts}
    cj_path = HERE / "glasgow_lod1.city.json"
    cj_path.write_text(json.dumps(cj, separators=(",", ":")))
    print(f"CityJSON: {len(cos_)} solids, {len(verts)} vertices -> {cj_path.name}")

    # ---- geometry ----
    import vtk
    from matplotlib import colormaps
    from matplotlib.colors import LogNorm
    norm = LogNorm(max(heights.min(), 3.0), heights.max())
    cmap = colormaps["turbo"]
    pts, cells, cols = vtk.vtkPoints(), vtk.vtkCellArray(), []
    for wid, p, h, src, tags in buildings:
        e = enu(p)
        n = len(e)
        base = pts.InsertNextPoint(e[0, 0], e[0, 1], 0.0)
        topids = []
        for i in range(n):
            if i:
                pts.InsertNextPoint(e[i, 0], e[i, 1], 0.0)
            topids.append(pts.InsertNextPoint(e[i, 0], e[i, 1], h))
        r, g, b, _ = cmap(norm(h))
        rgb = (int(r * 255), int(g * 255), int(b * 255), 255)
        for i in range(n):
            j = (i + 1) % n
            ids = vtk.vtkIdList()
            for k in (base + i, base + j, topids[j], topids[i]):
                ids.InsertNextId(k)
            cells.InsertNextCell(ids)
            cols.append(rgb)
        ids = vtk.vtkIdList()
        for k in reversed(topids):
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
    prop.SetLineWidth(0.5)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.85)

    cluster = np.vstack([enu(p).mean(axis=0) for _, p, _, _, _ in buildings[:12]])
    cx, cy = cluster[:, 0].mean(), cluster[:, 1].mean()
    span = max(2 * args.radius, float(heights.max()))
    cam = vtk.vtkCamera()
    cam.SetFocalPoint(cx, cy, 0.25 * span)
    cam.SetPosition(cx + 0.62 * span, cy - 0.95 * span, 0.58 * span)
    cam.SetViewUp(0, 0, 1)

    if args.interactive:
        from pyviz4d import Viewer4D
        viewer = Viewer4D(size=(W, H), bg_color=(0.12, 0.12, 0.14))
        viewer.add_actor(actor)
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
        viewer.start()
        return 0

    from pyviz4d import render_to_png
    render_to_png([actor], args.png, size=(W, H), camera=cam)
    import imageio.v2 as iio
    img = iio.imread(args.png).reshape(-1, 3)
    bg = np.abs(img.astype(int) - [38, 38, 38]).sum(1) <= 12
    print(f"PNG {args.png}: {W}x{H}, {pd.GetNumberOfPoints()} points, "
          f"{pd.GetNumberOfPolys()} faces, unique colours {len(np.unique(img, axis=0))}, "
          f"non-background {1 - bg.mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
