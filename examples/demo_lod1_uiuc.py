# Building data (c) Open City Model, BuildZero - ODbL 1.0
#   https://registry.opendata.aws/opencitymodel/  (footprints: Microsoft USBuildingFootprints)
#!/usr/bin/env python3
"""UIUC campus (Urbana-Champaign, Illinois) in pyviz4d, from Open City Model.

    uv run --extra geo python examples/demo_lod1_uiuc.py [--interactive]

Open City Model publishes LoD1 CityJSON for every US county on S3 (no AWS
account needed); Champaign County is Illinois/17019, ~68 MB across two files.
Each Building carries an extruded Solid, so unlike the OSM generators this one
needs no footprints or heights of its own -- it reads them, keeps the campus
bbox, and rebuilds the shells so the winding is outward-facing throughout
(shared machinery in ``_lod1.py``).  WGS84 (EPSG:4979) throughout; local ENU
metres for the geometry.

Reading the solids as they stand is not quite enough.  OCM's winding is not
uniform: almost all of these solids are already outward, but a few per cent
arrive inside-out, so rather than trust the source the script re-winds each
footprint ring outward (``ccw``).  A handful of rings also touch themselves --
the footprint simplification runs out to a vertex and straight back, enclosing
a sliver a few cm^2 in area -- which leaves the closed shell non-manifold;
those lobes are cut out (``despike``).

``--highlight ece`` paints the ECE Building magenta (matched to its OSM centre
within 9 m) and ``h`` toggles it live under ``--interactive``.  The colour is
written into the model's own cells, not into a second, coincident copy of the
walls and roof: two copies of one surface z-fight, which reads as the landmark
flashing between its base colour and magenta as the camera rotates.

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
from _lod1 import (
    M_PER_DEG_LAT,
    Building,
    ccw,
    despike,
    enu,
    lod1_scene,
    n_boundary_edges,
    oblique_camera,
    png_stats,
    viewpoint,
    write_cityjson,
)
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


def ingest(max_height):
    """Campus buildings as Building records (CCW lon/lat ring, height, source)."""
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
            buildings.append(Building(
                "ocm" + oid, ccw(p), h, src,
                {"footprint_source": a.get("fp_source"),
                 "footprint_area_m2": round(a.get("area", 0.0), 1),
                 "ubid": a.get("ubid"), "ocm_id": oid}))
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

    buildings.sort(key=lambda b: -b.height)
    heights = np.array([b.height for b in buildings])
    print(f"Open City Model {PREFIX}: {total} buildings in bbox, "
          f"{dropped} skipped, {len(buildings)} kept")
    print(f"height source {srcs}")
    print(f"height m: min {heights.min():.1f} med {np.median(heights):.1f} "
          f"max {heights.max():.1f}")
    print("tallest:")
    for b in buildings[:10]:
        print(f"  {b.height:6.1f} m  {len(b.ring):3d}-gon "
              f"{b.attrs['footprint_area_m2']:7.0f} m2  "
              f"ubid {b.attrs['ubid']}  [{b.source}]")

    write_cityjson(HERE / "uiuc_lod1.city.json",
                   "UIUC campus, Urbana-Champaign IL - LoD1 from Open City Model",
                   buildings, lon0, lat0)

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

    # ---- landmark highlight ----
    # The landmark is recoloured in the base actor's own cells (see
    # ``paint_highlight``).  Drawing a second, coincident copy of the walls and
    # roof -- as this did at first -- makes the two surfaces z-fight: whichever
    # of the two the rasteriser puts in front wins per pixel, and the boundary
    # between them slides around as the camera rotates, so the ECE Building
    # flashes orange (the base turbo colour) and magenta (the copy).  A polygon
    # offset only assigns a winner if exactly one of the pair is offset; here
    # the highlight mapper is offset away from the camera and the base mapper is
    # not offset at all, so there is no stable winner.  One copy, recoloured,
    # cannot fight with itself.  The pole is lines, not a surface, so it stays a
    # separate actor.
    HI_RGBA = (255, 0, 153, 255)                 # magenta: absent from turbo
    hi, hi_span = None, None
    if not args.no_highlight:
        lat, lon = LANDMARKS[args.highlight]
        want = np.array([(lon - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0)),
                         (lat - lat0) * M_PER_DEG_LAT])
        rings = [enu(b.ring, lon0, lat0) for b in buildings]
        near = [np.hypot(*(e.mean(0) - want)) for e in rings]
        k = int(np.argmin(near))
        b = buildings[k]
        print(f"highlight {args.highlight}: ubid {b.attrs['ubid']} "
              f"({b.height:.1f} m, {b.attrs['footprint_area_m2']:.0f} m2), "
              f"{near[k]:.0f} m from the landmark")

        hi_span = spans[k]                       # the cells to recolour
        cx, cy = rings[k].mean(0)                # a pole, so the pick reads
        ppt = vtk.vtkPoints()                    # from any angle and any zoom
        ppt.InsertNextPoint(cx, cy, b.height)
        ppt.InsertNextPoint(cx, cy, b.height + 0.35 * heights.max())
        pole = vtk.vtkCellArray()
        pl = vtk.vtkIdList()
        pl.InsertNextId(0)
        pl.InsertNextId(1)
        pole.InsertNextCell(pl)
        ppd = vtk.vtkPolyData()
        ppd.SetPoints(ppt)
        ppd.SetLines(pole)
        pm = vtk.vtkPolyDataMapper()
        pm.SetInputData(ppd)
        hi = vtk.vtkActor()                      # unlit, so it reads flat
        hi.SetMapper(pm)
        hp = hi.GetProperty()
        hp.SetColor(1.0, 0.0, 0.6)
        hp.SetLineWidth(2.5)
        hp.SetAmbient(1.0)
        hp.SetDiffuse(0.0)

    # ---- the `h` toggle ----
    # Recolour the landmark's cells inside the base actor's one scalar array,
    # rather than showing or hiding a second copy of its faces: the copy was the
    # flicker.  ``rgba.Modified()`` bumps the array's MTime, which is what makes
    # the mapper (DirectScalars) re-read the colours on the next Render.
    rgba = pd.GetCellData().GetScalars()
    hi_base = tuple(rgba.GetTuple4(hi_span[0])) if hi_span else None

    def paint_highlight(on):
        if hi_span is None:
            return
        c = HI_RGBA if on else hi_base
        for i in range(*hi_span):
            rgba.SetTuple4(i, *c)
        rgba.Modified()
        pd.Modified()

    if hi is not None:
        paint_highlight(True)

    # a flat campus needs a low, oblique eye: ~55 deg azimuth, ~25 deg elevation
    span = max(np.ptp(enu(np.array([[BBOX[0], BBOX[1]],
                                    [BBOX[2], BBOX[3]]]), lon0, lat0),
                      axis=0).max(),
               heights.max())
    cam = oblique_camera(0.0, 0.0, span, z_focus=0.04, dx=0.62, dy=-0.88, dz=0.34)
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
                    on = hi.GetVisibility() == 0
                    paint_highlight(on)
                    hi.SetVisibility(1 if on else 0)
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
    png_stats(args.png, (W, H), pd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
