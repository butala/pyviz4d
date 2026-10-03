# Building data (c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright
#!/usr/bin/env python3
"""James Watt Building area, University of Glasgow - LoD1 from OSM.

    uv run --extra geo python examples/demo_lod1_glasgow.py [--radius 350]

Geocodes the place with Nominatim, pulls OSM within +/- radius metres, extrudes
every building footprint to a LoD1 solid, writes CityJSON, and renders a still
with pyviz4d.  Add --interactive to open a Viewer4D window instead.  The
shared extrusion / CityJSON / render machinery is in ``_lod1.py``.
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from matplotlib import colormaps
from matplotlib.colors import LogNorm

from _lod1 import (Building, LEVEL_M, M_PER_DEG_LAT, ccw, clean, enu, height_of,
                   lod1_scene, n_boundary_edges, oblique_camera, png_stats,
                   viewpoint, write_cityjson)

HERE = Path(__file__).resolve().parents[1] / "data" / "glasgow"
HERE.mkdir(parents=True, exist_ok=True)
PLACE = "James Watt Building, Glasgow"
FALLBACK = (55.87165, -4.29135)          # University Avenue, Gilmorehill
UA = {"User-Agent": "pyviz4d-glasgow/0.1 (research)"}

# Floor counts by building type for tags OSM leaves without a height, and a
# church, which Glasgow's West End has rather more of than Pudong.
DEFAULT_LEVELS = {"yes": 3, "house": 2, "residential": 4, "apartments": 6,
                  "dormitory": 5, "school": 3, "university": 4, "college": 4,
                  "commercial": 4, "retail": 2, "hotel": 8, "hospital": 5,
                  "industrial": 2, "warehouse": 3, "construction": 3,
                  "office": 5, "civic": 4, "public": 4, "church": 8}
# A tenement is not a tower: 200 m is already generous here.
HEIGHT_RANGE = (2.0, 200.0)
LEVELS_RANGE = (0.5, 60.0)


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
        h, src = height_of(tags, DEFAULT_LEVELS,
                           height_range=HEIGHT_RANGE, levels_range=LEVELS_RANGE)
        buildings.append(Building(
            "b" + w.get("id"), ccw(p), h, src,
            {"osm_way": w.get("id"), "name": tags.get("name"),
             "building": tags.get("building", "yes")}))

    buildings.sort(key=lambda b: -b.height)
    heights = np.array([b.height for b in buildings])
    srcs = {}
    for b in buildings:
        srcs[b.source] = srcs.get(b.source, 0) + 1
    print(f"buildings {len(buildings)} | height source {srcs}")
    print(f"height m: max {heights.max():.1f} med {np.median(heights):.1f}")
    print("tallest:")
    for b in buildings[:8]:
        nm = b.attrs["name"] or "(unnamed)"
        print(f"  {b.height:6.1f} m  {nm[:46]:46s} {b.attrs['building']} [{b.source}]")

    write_cityjson(HERE / "glasgow_lod1.city.json",
                   "James Watt Building area, University of Glasgow - LoD1 (OSM)",
                   buildings, lon, lat)

    # ---- render ----
    norm = LogNorm(max(heights.min(), 3.0), heights.max())
    cmap = colormaps["turbo"]
    rgbs = [tuple(int(c * 255) for c in cmap(norm(b.height))[:3]) + (255,)
            for b in buildings]
    rings = [enu(b.ring, lon, lat) for b in buildings]
    pd, actor, spans = lod1_scene(rings, heights, rgbs, edge_width=0.5)
    n_open = n_boundary_edges(pd)             # 0 boundary edges = closed shells
    print(f"shells: {len(buildings)} buildings, {n_open} boundary edges "
          f"({'watertight' if n_open == 0 else 'NOT watertight'})")

    cluster = np.vstack([e.mean(axis=0) for e in rings[:12]])
    cx, cy = cluster[:, 0].mean(), cluster[:, 1].mean()
    span = max(2 * args.radius, float(heights.max()))
    cam = oblique_camera(cx, cy, span, z_focus=0.25, dx=0.62, dy=-0.95, dz=0.58)

    from pyviz4d import render_to_png
    if args.interactive:
        from pyviz4d import Viewer4D
        viewer = Viewer4D(size=(W, H), bg_color=(0.12, 0.12, 0.14))
        viewer.add_actor(actor)
        viewpoint(viewer.ren, cam)   # else the window opens on the default camera
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
        viewer.start()
        return 0

    render_to_png([actor], args.png, size=(W, H), camera=cam)
    png_stats(args.png, (W, H), pd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
