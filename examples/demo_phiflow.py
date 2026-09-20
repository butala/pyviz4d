import argparse
import numpy as np
import vtk
from tqdm import trange
from phi.flow import *
import matplotlib.pyplot as plt

from pyviz4d.viz import Viewer4D
from pyviz4d.volume import VolumeActor, IsosurfaceActor

def simulate_smoke(res=28, frames=200):
    print(f"Simulating {frames} frames of smoke plume at resolution {res} using pure NumPy (this might take a moment)...")

    # Simulation setup: Box is 100x200x100
    bounds = Box(x=(0, 100), y=(0, 200), z=(0, 100))
    velocity = StaggeredGrid(0, extrapolation.BOUNDARY, x=res, y=res*2, z=res, bounds=bounds)
    density = CenteredGrid(0, extrapolation.BOUNDARY, x=res, y=res*2, z=res, bounds=bounds)

    # Source at the bottom center
    source_sphere = Sphere(x=50, y=10, z=50, radius=10)
    inflow = 0.2 * CenteredGrid(source_sphere, extrapolation.BOUNDARY, x=res, y=res*2, z=res, bounds=bounds)

    density_arrays = []

    for i in trange(frames, desc="Simulating smoke frames"):
        density = advect.mac_cormack(density, velocity, dt=1) + inflow
        buoyancy = (density * vec(x=0, y=0.1, z=0)).at(velocity)

        # Advect velocity with high-order MacCormack to preserve vortices (reduced artificial diffusion)
        velocity = advect.mac_cormack(velocity, velocity, dt=1) + buoyancy

        # With open boundaries, the default solver is perfectly stable and fast.
        # We explicitly request 'scipy' CG solver which natively handles the sparse structure
        # up to res=32 on the CPU without triggering the pure NumPy overflow error.
        velocity, _ = fluid.make_incompressible(velocity, solve=Solve('CG', rel_tol=1e-3, abs_tol=1e-3, max_iterations=2000))

        # PhiFlow arrays are typically (x, y, z). We extract the numpy array.
        arr = density.values.numpy('x,y,z')
        density_arrays.append(arr)

    print("Simulation complete.")
    return density_arrays

def main():
    parser = argparse.ArgumentParser(description="PyViz4D PhiFlow Smoke Demo")
    parser.add_argument("--res", type=int, default=28, help="Grid resolution (X & Z axis). Y will be 2*res.")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to simulate.")
    args = parser.parse_args()

    densities = simulate_smoke(res=args.res, frames=args.frames)

    # 1. Initialize the 4D Viewer
    viewer = Viewer4D(bg_color=(0.1, 0.1, 0.12))

    # Enable Depth Peeling for order-independent transparency (Crucial for Isosurfaces!)
    viewer.ren.SetUseDepthPeeling(1)
    viewer.ren.SetOcclusionRatio(0.1)
    viewer.ren.SetMaximumNumberOfPeels(4)
    viewer.ren_win.SetAlphaBitPlanes(1)
    viewer.ren_win.SetMultiSamples(0)

    # 2. Setup the striking transparent IsosurfaceActor
    cmap = plt.get_cmap('viridis')
    iso_values = [0.05, 0.3, 0.8, 1.5, 2.5]
    colors = [cmap(v / 2.5)[:3] for v in iso_values]

    # Spacing mapping: Box size / Resolution
    # Box is (100, 200, 100), grid is (res, res*2, res)
    spacing = (100.0/args.res, 200.0/(args.res*2), 100.0/args.res)

    iso_actor = IsosurfaceActor(
        densities,
        spacing=spacing,
        iso_values=iso_values,
        colors=colors,
        opacity=0.35 # Glass-like transparency
    )
    viewer.add_actor(iso_actor)

    # 3. Add axes/bounding box
    cube_axes = vtk.vtkCubeAxesActor()
    cube_axes.SetBounds(0, 100, 0, 200, 0, 100)
    cube_axes.SetCamera(viewer.ren.GetActiveCamera())
    cube_axes.SetXTitle("X")
    cube_axes.SetYTitle("Y")
    cube_axes.SetZTitle("Z")

    for i in range(3):
        cube_axes.GetTitleTextProperty(i).SetColor(1, 1, 1)
        cube_axes.GetLabelTextProperty(i).SetColor(1, 1, 1)
    cube_axes.GetProperty().SetColor(0.5, 0.5, 0.5)

    viewer.add_actor(cube_axes)

    # 4. Position Camera
    cam = viewer.ren.GetActiveCamera()
    cam.SetPosition(250, 100, 300)
    cam.SetFocalPoint(50, 100, 50)
    cam.SetViewUp(0, 1, 0)
    viewer.ren.ResetCameraClippingRange()

    # 5. Add UI Controls for Time and Speed
    viewer.add_playback_ui(max_time=args.frames - 1, loop=True)

    print("Launching PyViz4D Viewer...")
    viewer.start(timer_interval_ms=16)

if __name__ == '__main__':
    main()
