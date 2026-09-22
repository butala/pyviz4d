# Building data (c) OpenStreetMap contributors, ODbL 1.0 - https://www.openstreetmap.org/copyright
#!/usr/bin/env python3
"""View the Li'an LoD1 CityJSON model interactively with pyviz4d.

    python data/lian/view_lian_lod1.py            # interactive window
    python data/lian/view_lian_lod1.py --png x.png  # offscreen check (no display)

Reads only the CityJSON (no network).  Mouse: rotate = left drag,
pan = middle/shift+left, zoom = right drag or scroll.  Press `q` to quit.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import vtk
from matplotlib import colormaps
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]
M_PER_DEG_LAT = 110574.0


def build_actor(cityjson_path, colour_by="height"):
    cj = json.loads(Path(cityjson_path).read_text())
    scale = cj["transform"]["scale"]
    tr = np.asarray(cj["transform"]["translate"], dtype=float)
    ll = np.asarray(cj["vertices"], dtype=float) * scale + tr   # lon, lat, z
    lat0, lon0 = ll[:, 1].mean(), ll[:, 0].mean()
    # local ENU metres about the model centre
    x = (ll[:, 0] - lon0) * M_PER_DEG_LAT * math.cos(math.radians(lat0))
    y = (ll[:, 1] - lat0) * M_PER_DEG_LAT
    xyz = np.c_[x, y, ll[:, 2]]

    objects = list(cj["CityObjects"].values())
    heights = np.array([o["attributes"]["height_m"] for o in objects])
    norm = Normalize(heights.min(), heights.max())
    cmap = colormaps["turbo"]

    pts = vtk.vtkPoints()
    cells = vtk.vtkCellArray()
    cell_colour = []
    object_id = []            # parallel to cells: index of the building
    for oi, o in enumerate(objects):
        h = o["attributes"]["height_m"]
        r, g, b, _ = cmap(norm(h))
        rgb = (int(r * 255), int(g * 255), int(b * 255), 255)
        for surface in o["geometry"][0]["boundaries"][0]:
            ring = surface if isinstance(surface[0], int) else surface[0]
            ids = vtk.vtkIdList()
            for vi in ring:
                ids.InsertNextId(pts.InsertNextPoint(*xyz[vi]))
            if ids.GetNumberOfIds() >= 3:
                cells.InsertNextCell(ids)
                cell_colour.append(rgb)
                object_id.append(oi)

    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    rgba = vtk.vtkUnsignedCharArray()
    rgba.SetNumberOfComponents(4)
    rgba.SetName("colors")
    for c in cell_colour:
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
    prop.SetEdgeColor(0.05, 0.05, 0.05)
    prop.SetLineWidth(0.7)
    prop.SetAmbient(0.35)
    prop.SetDiffuse(0.85)
    # per-building scalars exposed for picking / future colour ramps
    bheights = vtk.vtkFloatArray()
    bheights.SetName("height_m")
    for h in heights:
        bheights.InsertNextValue(h)
    return actor, pd, heights, norm, cmap, object_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cityjson", default=str(ROOT / "data" / "lian" / "lian_lod1.city.json"))
    ap.add_argument("--png", default=None, help="write a still instead of opening a window")
    ap.add_argument("--size", default="1200x900")
    args = ap.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))

    actor, pd, heights, norm, cmap, object_id = build_actor(args.cityjson)
    print(f"{pd.GetNumberOfPoints()} points, {pd.GetNumberOfPolys()} faces, "
          f"{len(heights)} buildings, height {heights.min():.0f}-{heights.max():.0f} m")

    if args.png:
        from pyviz4d import render_to_png
        cam = vtk.vtkCamera()
        cx = np.mean([pd.GetPoint(i)[0] for i in range(pd.GetNumberOfPoints())])
        cy = np.mean([pd.GetPoint(i)[1] for i in range(pd.GetNumberOfPoints())])
        span = 1300.0
        cam.SetFocalPoint(cx, cy, heights.max() * 0.5)
        cam.SetPosition(cx + 0.7 * span, cy - 0.95 * span, 0.7 * span)
        cam.SetViewUp(0, 0, 1)
        render_to_png([actor], args.png, size=(w, h), camera=cam)
        print("wrote", args.png)
        return 0

    from pyviz4d import Viewer4D
    viewer = Viewer4D(size=(w, h), bg_color=(0.12, 0.12, 0.14))
    viewer.add_actor(actor)                 # same actor as the offscreen path
    print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
    viewer.start()                          # blocks in the interactor loop
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
