# Building data (c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright
#!/usr/bin/env python3
"""Build a LoD1 CityJSON model of the Lingshui Li'an International Education
Innovation Pilot Zone (陵水黎安国际教育创新试验区, Hainan, China) from OSM.

Footprints: OSM ways tagged `building`. Height: `height` if present, else
`building:levels` * 3.0 m, else a per-type default. Geometry: one Solid per
building (bottom ring, one vertical quad per footprint edge, top ring).
Coordinates: WGS84 (OSM), local ENU metres for rendering, CityJSON transform
for storage.  Output is written under data/ (gitignored).
"""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib import colormaps

OUT = Path(__file__).resolve().parents[1] / "data" / "lian"
OUT.mkdir(parents=True, exist_ok=True)
ZONE_Q = "陵水黎安国际教育创新试验区"
UA = {"User-Agent": "pyviz4d-lod1/0.1 (research; contact: repo owner)"}
M_PER_DEG_LAT = 110574.0
LEVEL_M = 3.0
DEFAULT_LEVELS = {"yes": 2, "house": 2, "residential": 3, "apartments": 5,
                  "dormitory": 5, "school": 3, "university": 4, "college": 4,
                  "commercial": 3, "retail": 2, "hotel": 6, "hospital": 4,
                  "industrial": 2, "warehouse": 2, "construction": 3}


def zone_ring():
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": ZONE_Q, "format": "json", "limit": 1,
                             "polygon_geojson": 1}, headers=UA, timeout=30)
    r.raise_for_status()
    hit = r.json()[0]
    gj = hit["geojson"]
    rings = ([gj["coordinates"][0]] if gj["type"] == "Polygon"
             else [p[0] for p in gj["coordinates"]])
    ring = max(rings, key=len)  # outer ring of the largest polygon
    return np.asarray(ring, dtype=float), hit


def fetch_osm(bbox):
    lon0, lat0, lon1, lat1 = bbox
    r = requests.get("https://api.openstreetmap.org/api/0.6/map",
                     params={"bbox": f"{lon0},{lat0},{lon1},{lat1}"},
                     headers=UA, timeout=180)
    r.raise_for_status()
    return r.text


def parse(xml_text):
    root = ET.fromstring(xml_text)
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat")))
             for n in root.findall("node")}
    ways, rels = [], 0
    for w in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        refs = [nd.get("ref") for nd in w.findall("nd")]
        ways.append((w.get("id"), refs, tags))
    for rel in root.findall("relation"):
        if any(t.get("k") == "building" for t in rel.findall("tag")):
            rels += 1
    return nodes, ways, rels


def height_of(tags):
    h = tags.get("height")
    if h:
        try:
            v = float(str(h).split()[0].replace("m", ""))
            if 2.0 <= v <= 300.0:
                return v, "height"
        except ValueError:
            pass
    lv = tags.get("building:levels")
    if lv:
        try:
            v = float(str(lv).split(";")[0])
            if 0.5 <= v <= 100.0:
                return v * LEVEL_M, "levels"
        except ValueError:
            pass
    bt = tags.get("building", "yes")
    return DEFAULT_LEVELS.get(bt, 3) * LEVEL_M, "default:" + bt


def ring_area(ring):
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def main():
    ring, hit = zone_ring()
    lat0 = ring[:, 1].mean()
    lon0 = ring[:, 0].mean()
    pad = 0.004
    bbox = (ring[:, 0].min() - pad, ring[:, 1].min() - pad,
            ring[:, 0].max() + pad, ring[:, 1].max() + pad)
    print(f"zone: {hit['display_name']}")
    print(f"zone bbox lon {ring[:,0].min():.5f}..{ring[:,0].max():.5f} "
          f"lat {ring[:,1].min():.5f}..{ring[:,1].max():.5f}")

    nodes, ways, n_rel = parse(fetch_osm(bbox))
    inside = MplPath(ring)
    polys, skipped = [], 0
    for wid, refs, tags in ways:
        if "building" not in tags or len(refs) < 4:
            continue
        pts = [nodes[r] for r in refs if r in nodes]
        if len(pts) < 4:
            continue
        p = np.asarray(pts)
        if np.allclose(p[0], p[-1]):
            p = p[:-1]
        if len(p) < 3:
            skipped += 1
            continue
        # drop repeated/collinear-degenerate rings
        keep = np.r_[True, (np.abs(np.diff(p, axis=0)).sum(1) > 1e-9)]
        p = p[keep]
        if len(p) > 1 and np.allclose(p[0], p[-1]):
            p = p[:-1]
        if len(p) < 3 or not inside.contains_point(p.mean(axis=0)):
            continue
        polys.append((wid, p, tags))
    print(f"ways={len(ways)}  building relations(skipped, multipolygon)={n_rel}")
    print(f"buildings inside zone: {len(polys)}  (degenerate skipped {skipped})")

    # local ENU metres
    def enu(p):
        x = (p[:, 0] - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
        y = (p[:, 1] - lat0) * M_PER_DEG_LAT
        return np.c_[x, y]

    heights, srcs, solids, enu_polys = [], {}, [], []
    for wid, p, tags in polys:
        h, src = height_of(tags)
        heights.append(h)
        srcs[src] = srcs.get(src, 0) + 1
        enu_polys.append((wid, enu(p), h, tags, src))

    # ---- CityJSON (integer vertices via transform) ----
    scale = [1e-7, 1e-7, 0.001]
    translate = [round(lon0, 5), round(lat0, 5), 0.0]
    verts, vmap, co = [], {}, {}

    def vid(lon, lat, z):
        key = (round((lon - translate[0]) / scale[0]),
               round((lat - translate[1]) / scale[1]), round(z / scale[2]))
        if key not in vmap:
            vmap[key] = len(verts)
            verts.append([int(key[0]), int(key[1]), int(key[2])])
        return vmap[key]

    for wid, p, h, tags, src in enu_polys:
        # metres back to lon/lat for storage
        lon = p[:, 0] / (M_PER_DEG_LAT * math.cos(math.radians(lat0))) + lon0
        lat = p[:, 1] / M_PER_DEG_LAT + lat0
        n = len(lon)
        bot = [vid(lon[i], lat[i], 0.0) for i in range(n)]
        top = [vid(lon[i], lat[i], h) for i in range(n)]
        surfaces = [bot]                                   # ground
        for i in range(n):                                 # walls
            j = (i + 1) % n
            surfaces.append([bot[i], bot[j], top[j], top[i]])
        surfaces.append(top[::-1])                         # roof
        co["b" + str(wid)] = {
            "type": "Building",
            "attributes": {
                "height_m": round(h, 2),
                "height_source": src,
                "osm_way": wid,
                "building": tags.get("building", "yes"),
                "name": tags.get("name", None),
            },
            "geometry": [{"type": "Solid", "lod": "1", "boundaries": [surfaces]}],
        }

    cj = {"type": "CityJSON", "version": "1.1",
          "transform": {"scale": scale, "translate": translate},
          "metadata": {"referenceSystem": "https://www.opengis.net/def/crs/EPSG/0/4326",
                       "title": "Li'an International Education Innovation Pilot Zone, LoD1 (from OSM)"},
          "CityObjects": co, "vertices": verts}
    cj_path = OUT / "lian_lod1.city.json"
    cj_path.write_text(json.dumps(cj, separators=(",", ":")))
    print(f"CityJSON: {len(co)} solids, {len(verts)} vertices -> {cj_path} "
          f"({cj_path.stat().st_size/1e6:.2f} MB)")

    # ---- stats ----
    hs = np.array(heights)
    print(f"height source: {srcs}")
    print(f"height m: min {hs.min():.1f} med {np.median(hs):.1f} "
          f"max {hs.max():.1f} mean {hs.mean():.1f}")
    areas = np.array([ring_area(p) for _, p, _, _, _ in enu_polys])
    print(f"footprint m^2: min {areas.min():.0f} med {np.median(areas):.0f} "
          f"max {areas.max():.0f} total {areas.sum()/1e4:.2f} ha")
    ex = np.vstack([p for _, p, _, _, _ in enu_polys])
    print(f"extent m: x {ex[:,0].min():.0f}..{ex[:,0].max():.0f}  "
          f"y {ex[:,1].min():.0f}..{ex[:,1].max():.0f}")
    print(f"floor area (LoD1, m^2): {float((areas*hs/LEVEL_M).sum()):,.0f} "
          f"at {LEVEL_M} m/level equivalent")

    # ---- render with pyviz4d ----
    import vtk
    from pyviz4d import render_to_png
    cmap = colormaps["viridis"]
    norm = Normalize(hs.min(), hs.max())
    pts, cells, scal = vtk.vtkPoints(), vtk.vtkCellArray(), []
    for (wid, p, h, tags, src), area in zip(enu_polys, areas):
        n = len(p)
        base = pts.InsertNextPoint(p[0, 0], p[0, 1], 0.0)
        for i in range(1, n):
            pts.InsertNextPoint(p[i, 0], p[i, 1], 0.0)
        top = vtk.vtkIdList()
        for i in range(n):
            top.InsertNextId(pts.InsertNextPoint(p[i, 0], p[i, 1], h))
        for i in range(n):
            j = (i + 1) % n
            quad = vtk.vtkIdList()
            for k in (base + i, base + j, top.GetId(j), top.GetId(i)):
                quad.InsertNextId(k)
            cells.InsertNextCell(quad)
            scal.append(h)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    rgba = vtk.vtkUnsignedCharArray()
    rgba.SetNumberOfComponents(4)
    rgba.SetName("colors")
    for h in scal:
        r, g, b, _ = cmap(norm(h))
        rgba.InsertNextTuple4(int(r * 255), int(g * 255), int(b * 255), 255)
    pd.GetCellData().SetScalars(rgba)
    print("distinct cell colours:", len({tuple(rgba.GetTuple4(i))
                                        for i in range(rgba.GetNumberOfTuples())}))
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(pd)
    mapper.SetScalarModeToUseCellData()
    mapper.SetColorModeToDirectScalars()      # uchar RGBA -> literal colour
    mapper.SetScalarRange(0, 255)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetOpacity(1.0)
    prop.SetEdgeVisibility(1)                 # black edge lines read the LoD1 shape
    prop.SetEdgeColor(0.05, 0.05, 0.05)
    prop.SetLineWidth(0.6)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.85)
    # oblique view: default ResetCamera looks straight down, hiding the extrusion
    cam = vtk.vtkCamera()
    cx, cy = ex[:, 0].mean(), ex[:, 1].mean()
    span = max(np.ptp(ex[:, 0]), np.ptp(ex[:, 1]), float(hs.max()))
    cam.SetFocalPoint(cx, cy, hs.max() * 0.4)
    cam.SetPosition(cx + 0.75 * span, cy - 0.9 * span, 0.75 * span)
    cam.SetViewUp(0, 0, 1)
    png = OUT / "lian_lod1.png"
    render_to_png([actor], str(png), size=(1600, 1000), scale=1, camera=cam)
    print(f"PNG: {png} ({png.stat().st_size/1e3:.0f} kB), "
          f"{pd.GetNumberOfPoints()} points {pd.GetNumberOfPolys()} wall quads")
    return 0


if __name__ == "__main__":
    sys.exit(main())
