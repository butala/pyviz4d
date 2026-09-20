"""
Demo showing the decoupled Pipeline architecture:
1. Stage 1 (Simulate / Generate Data): Generates 3D data and writes binary VTK series (.vti + .pvd).
2. Stage 2 (Compose & Visualize): Reads the .pvd file lazily from disk into PyViz4D.
"""
import argparse
import os
import shutil
import numpy as np
import vtk
from tqdm import trange
import matplotlib.pyplot as plt

from pyviz4d import Viewer4D, VTKSeriesWriter, IsosurfaceSeriesActor


def run_simulation_and_export(output_dir: str, res: int = 32, frames: int = 60):
    """
    Stage 1: Pure simulation/data generation step.
    Computes a synthetic dynamic 3D vortex/wave field and writes binary .vti + .pvd to disk.
    """
    print(f"\n[Stage 1] Simulating & writing {frames} VTK binary frames to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    writer = VTKSeriesWriter(output_dir=output_dir, collection_name="density_field")

    x = np.linspace(-3, 3, res)
    y = np.linspace(-3, 3, res * 2)
    z = np.linspace(-3, 3, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    spacing = (6.0 / res, 12.0 / (res * 2), 6.0 / res)

    for t in trange(frames, desc="Exporting .vti files"):
        phase = t * 0.1
        # Dynamic rotating helical vortex field
        r = np.sqrt(X**2 + Z**2)
        theta = np.arctan2(Z, X)
        density = np.exp(-((r - 1.2 - 0.3 * np.sin(Y + phase))**2 + (Y - phase % 6 + 3)**2 * 0.2))
        density += 0.5 * np.exp(-((X - np.cos(phase))**2 + (Y * 0.5)**2 + (Z - np.sin(phase))**2))

        writer.write_image_data(density, timestep=float(t), filename_prefix="density", spacing=spacing)

    print(f"[Stage 1] Completed! PVD manifest written to: {writer.pvd_path}")
    return writer.pvd_path


def run_visualization_stage(pvd_path: str):
    """
    Stage 2: Visualization / composition step.
    Loads the .pvd on demand and interacts with the scene.
    """
    print(f"\n[Stage 2] Composing visualization from '{pvd_path}'...")
    viewer = Viewer4D(size=(1200, 800), bg_color=(0.08, 0.08, 0.1))

    # Transparency setup
    viewer.ren.SetUseDepthPeeling(1)
    viewer.ren.SetOcclusionRatio(0.1)
    viewer.ren.SetMaximumNumberOfPeels(4)
    viewer.ren_win.SetAlphaBitPlanes(1)
    viewer.ren_win.SetMultiSamples(0)

    # 1. Lazy-loaded Isosurface actor from disk
    cmap = plt.get_cmap('plasma')
    iso_values = [0.1, 0.3, 0.6, 0.9]
    colors = [cmap(v / max(iso_values))[:3] for v in iso_values]

    actor = IsosurfaceSeriesActor(
        pvd_path=pvd_path,
        iso_values=iso_values,
        colors=colors,
        opacity=0.45
    )
    viewer.add_actor(actor)

    # 2. Add bounding reference box
    cube_axes = vtk.vtkCubeAxesActor()
    cube_axes.SetBounds(0, 6, 0, 12, 0, 6)
    cube_axes.SetCamera(viewer.ren.GetActiveCamera())
    cube_axes.SetXTitle("X")
    cube_axes.SetYTitle("Y")
    cube_axes.SetZTitle("Z")
    for i in range(3):
        cube_axes.GetTitleTextProperty(i).SetColor(0.9, 0.9, 0.9)
        cube_axes.GetLabelTextProperty(i).SetColor(0.7, 0.7, 0.7)
    cube_axes.GetProperty().SetColor(0.3, 0.3, 0.3)
    viewer.add_actor(cube_axes)

    # 3. Position Camera
    cam = viewer.ren.GetActiveCamera()
    cam.SetPosition(18, 14, 18)
    cam.SetFocalPoint(3, 6, 3)
    cam.SetViewUp(0, 1, 0)
    viewer.ren.ResetCameraClippingRange()

    # 4. Attach playback UI
    viewer.add_playback_ui(max_time=actor.num_frames - 1, loop=True)

    print("Launching PyViz4D decoupled viewer...")
    viewer.start(timer_interval_ms=16)


def main():
    parser = argparse.ArgumentParser(description="PyViz4D VTK Series Export & Viz Demo")
    parser.add_argument("--output-dir", type=str, default="sim_output", help="Directory for VTK binary outputs")
    parser.add_argument("--res", type=int, default=32, help="Grid resolution")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames")
    parser.add_argument("--viz-only", action="store_true", help="Skip simulation and viz existing .pvd")
    args = parser.parse_args()

    pvd_file = os.path.join(args.output_dir, "density_field.pvd")

    if not args.viz_only or not os.path.exists(pvd_file):
        pvd_file = run_simulation_and_export(args.output_dir, res=args.res, frames=args.frames)

    run_visualization_stage(pvd_file)


if __name__ == '__main__':
    main()
