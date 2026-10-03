# Building data (c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright
#!/usr/bin/env python3
"""Build a LoD1 CityJSON model of the Lingshui Li'an International Education
Innovation Pilot Zone (陵水黎安国际教育创新试验区, Hainan, China) from OSM.

    uv run --extra geo python examples/demo_lod1_lian.py [--interactive]

Footprints: OSM ways tagged `building`, kept when their centre falls inside the
zone outline that Nominatim draws for the place.  Height: `height` if present,
else `building:levels` * 3.0 m, else a per-city default.  Geometry: one Solid
per building, closed (floor, one vertical quad per footprint edge, roof) --
shared machinery in ``_lod1.py``.  Coordinates: WGS84 (OSM), local ENU metres
for rendering, CityJSON transform for storage.  Output is written under
data/ (gitignored).
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath

from _lod1 import (Building, LEVEL_M, M_PER_DEG_LAT, ccw, clean, enu, height_of,
                   lod1_scene, n_boundary_edges, oblique_camera, ring_area,
                   viewpoint, write_cityjson)

OUT = Path(__file__).resolve().parents[1] / "data" / "lian"
OUT.mkdir(parents=True, exist_ok=True)
ZONE_Q = "陵水黎安国际教育创新试验区"
UA = {"User-Agent": "pyviz4d-lod1/0.1 (research; contact: repo owner)"}

# Floor counts by building type for tags OSM leaves without a height.  A Hainan
# campus is low-rise; the Shanghai table would overshoot badly here.
DEFAULT_LEVELS = {"yes": 2, "house": 2, "residential": 3, "apartments": 5,
                  "dormitory": 5, "school": 3, "university": 4, "college": 4,
                  "commercial": 3, "retail": 2, "hotel": 6, "hospital": 4,
                  "industrial": 2, "warehouse": 2, "construction": 3}
HEIGHT_RANGE = (2.0, 300.0)
LEVELS_RANGE = (0.5, 100.0)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=None, help="output PNG path")
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--interactive", action="store_true",
                    help="open a Viewer4D window instead of writing a PNG")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.lower().split("x"))

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
    buildings, skipped = [], 0
    for wid, refs, tags in ways:
        if "building" not in tags or len(refs) < 4:
            continue
        pts = [nodes[r] for r in refs if r in nodes]
        if len(pts) < 4:
            continue
        p = clean(np.asarray(pts))
        if len(p) < 3:
            skipped += 1
            continue
        if not inside.contains_point(p.mean(axis=0)):
            continue
        h, src = height_of(tags, DEFAULT_LEVELS,
                           height_range=HEIGHT_RANGE, levels_range=LEVELS_RANGE)
        buildings.append(Building(
            "b" + str(wid), ccw(p), h, src,
            {"osm_way": wid, "building": tags.get("building", "yes"),
             "name": tags.get("name", None)}))
    print(f"ways={len(ways)}  building relations(skipped, multipolygon)={n_rel}")
    print(f"buildings inside zone: {len(buildings)}  (degenerate skipped {skipped})")

    write_cityjson(OUT / "lian_lod1.city.json",
                   "Li'an International Education Innovation Pilot Zone, LoD1 (from OSM)",
                   buildings, lon0, lat0)

    # ---- stats ----
    heights = np.array([b.height for b in buildings])
    srcs = {}
    for b in buildings:
        srcs[b.source] = srcs.get(b.source, 0) + 1
    print(f"height source: {srcs}")
    print(f"height m: min {heights.min():.1f} med {np.median(heights):.1f} "
          f"max {heights.max():.1f} mean {heights.mean():.1f}")
    rings = [enu(b.ring, lon0, lat0) for b in buildings]
    areas = np.array([ring_area(e) for e in rings])    # CCW, so positive
    print(f"footprint m^2: min {areas.min():.0f} med {np.median(areas):.0f} "
          f"max {areas.max():.0f} total {areas.sum()/1e4:.2f} ha")
    ex = np.vstack(rings)
    print(f"extent m: x {ex[:,0].min():.0f}..{ex[:,0].max():.0f}  "
          f"y {ex[:,1].min():.0f}..{ex[:,1].max():.0f}")
    print(f"floor area (LoD1, m^2): {float((areas*heights/LEVEL_M).sum()):,.0f} "
          f"at {LEVEL_M} m/level equivalent")

    # ---- render ----
    norm = Normalize(heights.min(), heights.max())
    cmap = colormaps["viridis"]
    rgbs = [tuple(int(c * 255) for c in cmap(norm(b.height))[:3]) + (255,)
            for b in buildings]
    pd, actor, spans = lod1_scene(rings, heights, rgbs,
                                  edge_rgb=(0.05, 0.05, 0.05), edge_width=0.6)
    n_open = n_boundary_edges(pd)             # 0 boundary edges = closed shells
    print(f"shells: {len(buildings)} buildings, {n_open} boundary edges "
          f"({'watertight' if n_open == 0 else 'NOT watertight'})")
    print("distinct cell colours:", len(set(rgbs)))

    # oblique view: default ResetCamera looks straight down, hiding the extrusion
    cx, cy = ex[:, 0].mean(), ex[:, 1].mean()
    span = max(np.ptp(ex[:, 0]), np.ptp(ex[:, 1]), float(heights.max()))
    cam = oblique_camera(cx, cy, span,
                         z_focus=0.4 * heights.max() / span,
                         dx=0.75, dy=-0.9, dz=0.75)

    from pyviz4d import render_to_png
    if args.interactive:
        from pyviz4d import Viewer4D
        viewer = Viewer4D(size=(W, H), bg_color=(0.12, 0.12, 0.14))
        viewer.add_actor(actor)
        viewpoint(viewer.ren, cam)   # else the window opens on the default camera
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
        viewer.start()
        return 0

    png = Path(args.png) if args.png else OUT / "lian_lod1.png"
    render_to_png([actor], str(png), size=(W, H), scale=1, camera=cam)
    print(f"PNG: {png} ({png.stat().st_size/1e3:.0f} kB), "
          f"{pd.GetNumberOfPoints()} points "
          f"{pd.GetNumberOfPolys()} faces (walls + roof + floor)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
