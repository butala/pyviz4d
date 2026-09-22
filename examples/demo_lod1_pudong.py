# Building data (c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright
#!/usr/bin/env python3
"""Lujiazui (Pudong, Shanghai) skyline in pyviz4d, from OSM footprints + heights.

    python data/pudong/make_pudong.py [--png out.png]

Writes a CityJSON LoD1 model, an offscreen PNG, and prints the tallest
buildings.  Heights are real where OSM has them (`height`), else
`building:levels` x 3 m, else a per-type default.  WGS84 throughout;
local ENU metres for geometry.
"""
import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
import vtk
from matplotlib import colormaps
from matplotlib.colors import LogNorm

HERE = Path(__file__).resolve().parents[1] / "data" / "pudong"
HERE.mkdir(parents=True, exist_ok=True)
# Lujiazui: Bund bend to Century Ave, Oriental Pearl through Shanghai Tower
BBOX = (121.4920, 31.2280, 121.5160, 31.2450)          # lon0, lat0, lon1, lat1
UA = {"User-Agent": "pyviz4d-pudong/0.1 (research)"}
M_PER_DEG_LAT = 110574.0
LEVEL_M = 3.0
DEFAULT_LEVELS = {"yes": 3, "house": 2, "residential": 6, "apartments": 8,
                  "dormitory": 5, "school": 3, "university": 4, "college": 4,
                  "commercial": 4, "retail": 2, "hotel": 10, "hospital": 5,
                  "industrial": 2, "warehouse": 2, "construction": 4,
                  "office": 8, "civic": 4, "public": 4}


def fetch():
    cache = HERE / "lujiazui.osm"
    if cache.exists():
        return cache.read_text()
    r = requests.get("https://api.openstreetmap.org/api/0.6/map",
                     params={"bbox": ",".join(str(v) for v in BBOX)},
                     headers=UA, timeout=300)
    r.raise_for_status()
    cache.write_text(r.text)
    return r.text


def height_of(tags):
    h = tags.get("height")
    if h:
        try:
            v = float(str(h).split()[0].replace("m", ""))
            if 2.0 <= v <= 700.0:
                return v, "height"
        except ValueError:
            pass
    lv = tags.get("building:levels")
    if lv:
        try:
            v = float(str(lv).split(";")[0])
            if 0.5 <= v <= 200.0:
                return v * LEVEL_M, "levels"
        except ValueError:
            pass
    bt = tags.get("building", "yes")
    return DEFAULT_LEVELS.get(bt, 3) * LEVEL_M, "default:" + bt


def clean(p):
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    keep = np.r_[True, (np.abs(np.diff(p, axis=0)).sum(1) > 1e-9)]
    p = p[keep]
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=str(HERE / "pudong.png"))
    ap.add_argument("--size", default="1600x1000")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.lower().split("x"))

    root = ET.fromstring(fetch())
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat")))
             for n in root.findall("node")}
    lat0 = (BBOX[1] + BBOX[3]) / 2
    lon0 = (BBOX[0] + BBOX[2]) / 2
    coslat = math.cos(math.radians(lat0))

    ways = rels = 0
    buildings = []
    for w in root.findall("way"):
        ways += 1
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        if "building" not in tags:
            continue
        pts = [nodes[r] for r in (nd.get("ref") for nd in w.findall("nd"))
               if r in nodes]
        if len(pts) < 4:
            continue
        p = clean(np.asarray(pts))
        if len(p) < 3:
            continue
        h, src = height_of(tags)
        buildings.append((w.get("id"), p, h, src, tags))
    for rel in root.findall("relation"):
        if any(t.get("k") == "building" for t in rel.findall("tag")):
            rels += 1

    def enu(p):
        return np.c_[(p[:, 0] - lon0) * M_PER_DEG_LAT * coslat,
                     (p[:, 1] - lat0) * M_PER_DEG_LAT]

    buildings.sort(key=lambda b: -b[2])
    heights = np.array([b[2] for b in buildings])
    srcs = {}
    for b in buildings:
        srcs[b[3]] = srcs.get(b[3], 0) + 1
    print(f"OSM ways {ways}, building multipolygon relations skipped {rels}")
    print(f"buildings {len(buildings)} | height source {srcs}")
    print(f"height m: max {heights.max():.0f} med {np.median(heights):.0f}")
    print("tallest:")
    for wid, p, h, src, tags in buildings[:12]:
        name = tags.get("name") or tags.get("name:en") or "(unnamed)"
        print(f"  {h:6.1f} m  {name[:44]:44s} {tags.get('building')} [{src}]")

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

    for wid, p, h, src, tags in buildings:
        lon = p[:, 0]
        lat = p[:, 1]
        n = len(p)
        bot = [vid(lon[i], lat[i], 0.0) for i in range(n)]
        top = [vid(lon[i], lat[i], h) for i in range(n)]
        surfaces = [bot] + [[bot[i], bot[(i + 1) % n],
                             top[(i + 1) % n], top[i]] for i in range(n)] + [top[::-1]]
        cos["b" + wid] = {"type": "Building",
                          "attributes": {"height_m": round(h, 2), "height_source": src,
                                         "osm_way": wid, "name": tags.get("name"),
                                         "name_en": tags.get("name:en"),
                                         "building": tags.get("building", "yes")},
                          "geometry": [{"type": "Solid", "lod": "1",
                                        "boundaries": [surfaces]}]}
    cj = {"type": "CityJSON", "version": "1.1",
          "transform": {"scale": scale, "translate": translate},
          "metadata": {"referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4326",
                       "title": "Lujiazui, Pudong, Shanghai - LoD1 from OSM"},
          "CityObjects": cos, "vertices": verts}
    cj_path = HERE / "pudong_lod1.city.json"
    cj_path.write_text(json.dumps(cj, separators=(",", ":")))
    print(f"CityJSON: {len(cos)} solids, {len(verts)} vertices, "
          f"{cj_path.stat().st_size/1e6:.2f} MB")

    # ---- render ----
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
        ids = vtk.vtkIdList()                     # roof
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
    prop.SetLineWidth(0.4)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.85)

    # frame the supertall cluster, not the whole bbox: the towers are a small
    # part of the 2.3 x 1.9 km window and look lost when the bbox is fitted
    cluster = np.vstack([enu(p).mean(axis=0) for _, p, _, _, _ in buildings[:20]])
    cx, cy = cluster[:, 0].mean(), cluster[:, 1].mean()
    span = 0.72 * max(heights.max(),
                      np.ptp(enu(np.array([[BBOX[0], BBOX[1]],
                                           [BBOX[2], BBOX[3]]])), axis=0).max())
    cam = vtk.vtkCamera()
    cam.SetFocalPoint(cx, cy, 0.30 * span)
    cam.SetPosition(cx + 0.62 * span, cy - 0.95 * span, 0.58 * span)
    cam.SetViewUp(0, 0, 1)
    # NOTE: only the camera's *direction*, focal point and view-up survive --
    # render_to_png() ends with ren.ResetCamera(), which recomputes the distance
    # from the current view angle and therefore cancels any SetPosition radius
    # or pre-applied Zoom().  For a tighter frame, use the interactive viewer
    # (mouse zoom) or move the focal point; a `zoom=` kwarg on render_to_png
    # would fix this properly.

    from pyviz4d import render_to_png
    render_to_png([actor], args.png, size=(W, H), camera=cam)

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
