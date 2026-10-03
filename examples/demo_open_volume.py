"""Open scientific volumes, rendered properly.

    uv run --extra geo python examples/demo_open_volume.py --dataset aneurism
    uv run --extra geo python examples/demo_open_volume.py --dataset foot --interactive

Three iconic volumes from `klacansky.com/open-scivis-datasets`, which collects
public domain scientific scans and makes them available as raw arrays.  They
are the classic material for judging volume rendering because their structure
is strong enough to survive any reasonable transfer function:

* ``aneurism`` -- rotational C-arm angiogram of the arteries of a head, with a
  contrast agent in the blood and an aneurism present.  The vessels are the
  bright end of the scalar range, so this is a tree of light.
* ``foot``     -- C-arm scan of a human foot, *tissue and bone*.  The two
  materials sit at different densities, which is the textbook case for a
  two-material transfer function: translucent skin over solid bone.
* ``bonsai``   -- microCT of a bonsai tree.

17 MB each, downloaded once into ``data/volumes/``.

Why the transfer functions here are shaped the way they are
----------------------------------------------------------
Two rules, both learned the hard way on the smoke plume in
``demo_phiflow.py``:

1. **Colour points live in the data domain.**  ``vtkColorTransferFunction``
   maps *scalar values* to colour, so the x's must span the data -- 0..255
   here -- not 0..1.  Getting this wrong silently maps the whole volume onto
   one colour.
2. **Flat bands, and emissive.**  A smooth ramp averages along the ray into
   grey (measured saturation 0.06), and diffuse+specular volume shading adds
   white to every sample on top of that.  Held colour bands rendered
   unlit give the saturated, poster-like look these scans want.

Neither of these is a matter of taste; they are what separates "a saturated
object" from "a grey cloud".
"""
import argparse
from pathlib import Path

import numpy as np
import vtk

from pyviz4d import Viewer4D, VolumeActor, render_to_png, viewpoint

HERE = Path(__file__).resolve().parents[1] / "data" / "volumes"
BASE = "http://klacansky.com/open-scivis-datasets"
SHAPE = (256, 256, 256)

# Transfer functions, as (scalar, r, g, b) and (scalar, opacity) over 0..255.
# Held over a stretch so material in a band renders at that one colour.
PRESETS = {
    "aneurism": dict(
        about="arteries of a head, contrast agent, an aneurism",
        # Floor at 42 rather than 28: below that is scan noise, which hung off
        # the tree as dark wisps.  Then deep amber -> gold -> white, so the
        # vessel walls read warm and only the contrast-filled cores blow out.
        color=[(0, 0.02, 0.01, 0.03), (45, 0.02, 0.01, 0.03),
               (48, 0.42, 0.10, 0.03), (95, 0.42, 0.10, 0.03),
               (98, 0.86, 0.34, 0.06), (145, 0.86, 0.34, 0.06),
               (148, 1.00, 0.72, 0.20), (190, 1.00, 0.72, 0.20),
               (193, 1.00, 1.00, 1.00), (255, 1.00, 1.00, 1.00)],
        opacity=[(0, 0.0), (45, 0.0), (49, 0.30), (95, 0.58),
                 (150, 0.90), (255, 1.0)]),
    "foot": dict(
        about="human foot -- translucent tissue over solid bone",
        color=[(0, 0.02, 0.02, 0.05), (55, 0.02, 0.02, 0.05),
               (58, 0.45, 0.12, 0.30), (95, 0.45, 0.12, 0.30),
               (98, 0.90, 0.45, 0.25), (150, 0.90, 0.45, 0.25),
               (153, 1.00, 0.96, 0.88), (255, 1.00, 1.00, 1.00)],
        opacity=[(0, 0.0), (55, 0.0), (60, 0.06), (110, 0.14),
                 (145, 0.55), (185, 0.95), (255, 1.0)]),
    "bonsai": dict(
        about="microCT of a bonsai tree",
        color=[(0, 0.02, 0.03, 0.02), (45, 0.02, 0.03, 0.02),
               (48, 0.35, 0.14, 0.05), (95, 0.35, 0.14, 0.05),
               (98, 0.25, 0.65, 0.12), (150, 0.25, 0.65, 0.12),
               (153, 0.85, 0.95, 0.35), (255, 1.00, 1.00, 1.00)],
        opacity=[(0, 0.0), (45, 0.0), (50, 0.10), (100, 0.35),
                 (150, 0.85), (255, 1.0)]),
}


def fetch(name):
    """Download one raw volume once (17 MB) into data/volumes/."""
    HERE.mkdir(parents=True, exist_ok=True)
    fname = f"{name}_256x256x256_uint8.raw"
    path = HERE / fname
    if not path.exists():
        import requests
        url = f"{BASE}/{name}/{fname}"
        print(f"downloading {url}")
        r = requests.get(url, headers={"User-Agent": "pyviz4d-volume/0.1 (research)"},
                         timeout=300)
        r.raise_for_status()
        path.write_bytes(r.content)
    return np.fromfile(path, dtype=np.uint8).reshape(SHAPE).astype(np.float32)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=sorted(PRESETS), default="aneurism")
    p.add_argument("--png", default=None, help="default docs/volume_<dataset>.png")
    p.add_argument("--size", nargs=2, type=int, default=[1100, 900])
    p.add_argument("--zoom", type=float, default=1.85)
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    spec = PRESETS[args.dataset]
    vol = fetch(args.dataset)
    print(f"{args.dataset}: {spec['about']} -- {vol.shape}, "
          f"scalar range {vol.min():.0f}..{vol.max():.0f}, "
          f"p50 {np.median(vol):.0f} p99 {np.percentile(vol, 99):.0f}")

    actor = VolumeActor([vol], spacing=(1.0, 1.0, 1.0),
                        color_points=spec["color"], opacity_points=spec["opacity"],
                        sample_distance=0.4)
    # Part-lit, not emissive.  Emissive was right for the smoke plume (shading
    # a *gradient* greys it out) but these are solid tubes and shells: without
    # a lighting term they read as flat ribbons.  Ambient carries the colour,
    # diffuse gives the roundness, and a sharp specular gives the highlight the
    # reference image has.
    actor.prop.SetAmbient(0.55)
    actor.prop.SetDiffuse(0.55)
    actor.prop.SetSpecular(0.40)
    actor.prop.SetSpecularPower(42)

    cam = vtk.vtkCamera()
    cam.SetFocalPoint(128, 128, 128)
    cam.SetPosition(128 + 340, 128 - 420, 128 + 300)
    cam.SetViewUp(0, 0, 1)

    out = args.png or str(Path("docs") / f"volume_{args.dataset}.png")
    if args.interactive:
        viewer = Viewer4D(size=(args.size[0], args.size[1]),
                          bg_color=(0.0, 0.0, 0.0))
        viewer.ren_win.SetMultiSamples(8)
        viewer.add_actor(actor.actor)
        viewpoint(viewer.ren, cam)
        print("window: left-drag rotate, middle/shift-drag pan, scroll zoom, q quit")
        viewer.start()
        return 0

    render_to_png([actor.actor], out, size=(args.size[0], args.size[1]),
                  camera=cam, bg_color=(0.0, 0.0, 0.0), zoom=args.zoom)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
