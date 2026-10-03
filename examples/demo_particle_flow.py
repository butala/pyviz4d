"""A 4-D particle simulation, live: thousands of particles, for ever.

    uv run --extra dev python examples/demo_particle_flow.py

Not a pre-rendered animation.  40 000 particles are advected through the
contact-tank flow with RK4 plus an eddy diffusivity, and any that leave are
reborn at the inlet -- so the cloud reaches a statistical steady state and the
simulation simply runs, at the frame rate of the window, for as long as you
let it.  There is no last frame and nothing is cached.

Semi-transparent gaussian splats, so forty thousand of them accumulate into a
cloud you can see through rather than a solid mass.  Coloured by how long each
particle has been in the tank: cool where fluid is young and moving, warm in
the stagnant pools behind the baffles.

Left-drag rotates while it runs.  The physics is one vectorised numpy pass per
frame -- tens of thousands of particles cost a few milliseconds, which is why
this can be endless and interactive at the same time.
"""
import argparse

import numpy as np
import vtk
from demo_contact_tank import H, L, W, tank_edges, velocity_at

from pyviz4d import ParticleCloudActor, Viewer4D, render_to_png, viewpoint


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--particles", type=int, default=40000)
    p.add_argument("--dt", type=float, default=0.045)
    p.add_argument("--diffusivity", type=float, default=0.02)
    p.add_argument("--opacity", type=float, default=0.085,
                   help="per-splat alpha; low accumulates into a see-through cloud")
    p.add_argument("--splat-scale", type=float, default=1.0,
                   help="splat radius multiplier (0.05 m at 1.0)")
    p.add_argument("--color-by", choices=["age", "speed"], default="age")
    p.add_argument("--png", default="docs/particle_flow.png",
                   help="still only in --steps mode")
    p.add_argument("--warmup", type=int, default=150,
                   help="steps integrated before the window opens")
    p.add_argument("--steps", type=int, default=0,
                   help=">0: integrate this many steps and write a still instead "
                        "of opening the endless window")
    p.add_argument("--size", nargs=2, type=int, default=[1400, 900])
    args = p.parse_args()

    bounds = ((0.0, L), (0.0, W), (0.0, H))
    inlet = ((0.0, 1.2), (0.3, W - 0.3), (0.15 * H, 0.85 * H))

    cloud = ParticleCloudActor(velocity_at, bounds, inlet,
                               n=args.particles, dt=args.dt,
                               kappa=args.diffusivity, color_by=args.color_by,
                               splat_scale=args.splat_scale,
                               opacity=args.opacity, seed=0)
    # Warm up, then calibrate the colour scale off the cloud that actually
    # develops.  A fixed age_max of 1 clamps every particle to the hot end and
    # the picture is one flat colour (tried); the residence times here run to
    # several time units, so the scale has to come from the simulation.
    for _ in range(args.warmup):
        cloud.update(0.0)
    cloud.age_max = float(np.percentile(cloud.age, 95)) or 1.0
    print(f"{args.particles} particles  dt {args.dt}  kappa {args.diffusivity}  "
          f"alpha {args.opacity}  age scale 0..{cloud.age_max:.2f}")

    cam = vtk.vtkCamera()
    cam.SetFocalPoint(0.5 * L, 0.5 * W, 0.45 * H)
    cam.SetPosition(0.5 * L + 1.1 * L, -0.85 * L, 0.55 * L)
    cam.SetViewUp(0, 0, 1)

    if args.steps:
        for i in range(args.steps):
            cloud.update(float(i))
            if (i + 1) % 50 == 0:
                print(f"  step {i+1:5d}  age p50 {np.median(cloud.age):.2f}")
        render_to_png([cloud.actor, tank_edges()], args.png,
                      size=(args.size[0], args.size[1]), camera=cam,
                      bg_color=(0.01, 0.01, 0.02), zoom=1.3)
        print(f"wrote {args.png} after {args.steps} steps")
        return 0

    viewer = Viewer4D(size=(args.size[0], args.size[1]),
                      bg_color=(0.01, 0.01, 0.02))
    viewer.ren_win.SetMultiSamples(4)
    viewer.add_actor(cloud)
    viewer.add_actor(tank_edges())
    viewpoint(viewer.ren, cam)
    print("running for ever -- left-drag rotate, middle/shift-drag pan, "
          "scroll zoom, q quit")
    viewer.start(timer_interval_ms=16)     # ~60 fps: 4 RK4 stages each tick
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
