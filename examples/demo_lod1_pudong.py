# Building data (c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright
#!/usr/bin/env python3
"""Lujiazui (Pudong, Shanghai) skyline in pyviz4d, from OSM footprints + heights.

    uv run --extra geo python examples/demo_lod1_pudong.py [--interactive]

Writes a CityJSON LoD1 model, an offscreen PNG, and prints the tallest
buildings.  Heights are real where OSM has them (`height`), else
`building:levels` x 3 m, else a per-city default table.  WGS84 throughout;
local ENU metres for geometry.  The shared extrusion / CityJSON / render
machinery is in ``_lod1.py``; what is local to Lujiazui is the bbox, the
height table and the framing of the supertall cluster.
"""
import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests
from matplotlib import colormaps
from matplotlib.colors import LogNorm

from _lod1 import (Building, LEVEL_M, ccw, clean, enu, height_of, lod1_scene,
                   n_boundary_edges, oblique_camera, png_stats, viewpoint,
                   write_cityjson)

HERE = Path(__file__).resolve().parents[1] / "data" / "pudong"
HERE.mkdir(parents=True, exist_ok=True)
# Lujiazui: Bund bend to Century Ave, Oriental Pearl through Shanghai Tower
BBOX = (121.4920, 31.2280, 121.5160, 31.2450)          # lon0, lat0, lon1, lat1
UA = {"User-Agent": "pyviz4d-pudong/0.1 (research)"}

# Floor counts by building type for tags OSM leaves without a height.  A
# Pudong block is not a Hainan campus, so this table is per city.
DEFAULT_LEVELS = {"yes": 3, "house": 2, "residential": 6, "apartments": 8,
                  "dormitory": 5, "school": 3, "university": 4, "college": 4,
                  "commercial": 4, "retail": 2, "hotel": 10, "hospital": 5,
                  "industrial": 2, "warehouse": 2, "construction": 4,
                  "office": 8, "civic": 4, "public": 4}
# Shanghai has supertalls; a 700 m ceiling drops typos without clipping them.
HEIGHT_RANGE = (2.0, 700.0)
LEVELS_RANGE = (0.5, 200.0)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default=str(HERE / "pudong.png"))
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--interactive", action="store_true",
                    help="open a Viewer4D window instead of writing a PNG")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.lower().split("x"))

    root = ET.fromstring(fetch())
    nodes = {n.get("id"): (float(n.get("lon")), float(n.get("lat")))
             for n in root.findall("node")}
    lat0 = (BBOX[1] + BBOX[3]) / 2
    lon0 = (BBOX[0] + BBOX[2]) / 2

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
        h, src = height_of(tags, DEFAULT_LEVELS,
                           height_range=HEIGHT_RANGE, levels_range=LEVELS_RANGE)
        buildings.append(Building(
            "b" + w.get("id"), ccw(p), h, src,
            {"osm_way": w.get("id"), "name": tags.get("name"),
             "name_en": tags.get("name:en"), "building": tags.get("building", "yes")}))
    for rel in root.findall("relation"):
        if any(t.get("k") == "building" for t in rel.findall("tag")):
            rels += 1

    buildings.sort(key=lambda b: -b.height)
    heights = np.array([b.height for b in buildings])
    srcs = {}
    for b in buildings:
        srcs[b.source] = srcs.get(b.source, 0) + 1
    print(f"OSM ways {ways}, building multipolygon relations skipped {rels}")
    print(f"buildings {len(buildings)} | height source {srcs}")
    print(f"height m: max {heights.max():.0f} med {np.median(heights):.0f}")
    print("tallest:")
    for b in buildings[:12]:
        name = b.attrs["name"] or b.attrs["name_en"] or "(unnamed)"
        print(f"  {b.height:6.1f} m  {name[:44]:44s} {b.attrs['building']} [{b.source}]")

    write_cityjson(HERE / "pudong_lod1.city.json",
                   "Lujiazui, Pudong, Shanghai - LoD1 from OSM", buildings,
                   lon0, lat0)

    # ---- render ----
    norm = LogNorm(max(heights.min(), 3.0), heights.max())
    cmap = colormaps["turbo"]
    rgbs = [tuple(int(c * 255) for c in cmap(norm(b.height))[:3]) + (255,)
            for b in buildings]
    pd, actor, spans = lod1_scene([enu(b.ring, lon0, lat0) for b in buildings],
                                  heights, rgbs)
    n_open = n_boundary_edges(pd)             # 0 boundary edges = closed shells
    print(f"shells: {len(buildings)} buildings, {n_open} boundary edges "
          f"({'watertight' if n_open == 0 else 'NOT watertight'})")

    # frame the supertall cluster, not the whole bbox: the towers are a small
    # part of the 2.3 x 1.9 km window and look lost when the bbox is fitted
    rings = [enu(b.ring, lon0, lat0) for b in buildings]
    cluster = np.vstack([e.mean(axis=0) for e in rings[:20]])
    cx, cy = cluster[:, 0].mean(), cluster[:, 1].mean()
    span = 0.72 * max(heights.max(),
                      np.ptp(enu(np.array([[BBOX[0], BBOX[1]],
                                           [BBOX[2], BBOX[3]]]), lon0, lat0),
                             axis=0).max())
    cam = oblique_camera(cx, cy, span, z_focus=0.30, dx=0.62, dy=-0.95, dz=0.58)
    # NOTE: only the camera's *direction* and view-up survive --
    # render_to_png() ends with ren.ResetCamera(), which recomputes the distance
    # from the current view angle and therefore cancels any SetPosition radius
    # or pre-applied Zoom().  For a tighter frame, use the interactive viewer
    # (mouse zoom) or move the focal point.

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
