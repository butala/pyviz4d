import argparse
import numpy as np
import vtk
from pyviz4d.viz import EarthViewer4D, TemporalActor
from pyviz4d.earth import WGS84
from pyviz4d.spherical_grid import spherical_grid_actor

class Satellite(TemporalActor):
    def __init__(self, radius, speed):
        self.radius = radius
        self.speed = speed

        # Create a sphere to represent the satellite
        source = vtk.vtkSphereSource()
        source.SetRadius(200.0)
        source.SetThetaResolution(20)
        source.SetPhiResolution(20)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.0, 0.0) # Red

        super().__init__(actor)

    def update(self, current_time: float):
        angle = current_time * self.speed
        x = self.radius * np.cos(angle)
        y = self.radius * np.sin(angle)
        z = np.sin(angle * 0.5) * 2000

        # Update physical position of the actor in the world
        self.actor.SetPosition(x, y, z)


def main():
    parser = argparse.ArgumentParser(description="PyViz4D Earth Demo")
    parser.add_argument("--record", action="store_true", help="Record to output.mp4")
    parser.add_argument("--save-frames", action="store_true", help="Save individual frames to 'frames/' directory")
    parser.add_argument("--max-frames", type=int, default=300, help="Number of frames to record (default: 300)")
    args = parser.parse_args()

    # 1. Initialize the 4D Viewer with the WGS84 Earth loaded
    viewer = EarthViewer4D()

    # Enable recording if requested
    if args.record or args.save_frames:
        video_path = "output.mp4" if args.record else None
        frames_dir = "frames" if args.save_frames else None
        print(f"Recording enabled: max_frames={args.max_frames}")
        if video_path: print(f" - Video output: {video_path}")
        if frames_dir: print(f" - Frames directory: {frames_dir}/")

        viewer.enable_recording(
            video_path=video_path,
            frames_dir=frames_dir,
            fps=60,
            max_frames=args.max_frames
        )

    # 2. Create the satellite and add it as a TemporalActor
    r = WGS84.a / 1e3 + 2000
    sat = Satellite(radius=r, speed=0.05)
    viewer.add_actor(sat)

    # 4. We can easily add standard VTK scientific tools (like a bounding box)
    cube_axes = vtk.vtkCubeAxesActor()
    cube_axes.SetBounds(-10000, 10000, -10000, 10000, -10000, 10000)
    cube_axes.SetCamera(viewer.ren.GetActiveCamera())
    cube_axes.SetXTitle("X (km)")
    cube_axes.SetYTitle("Y (km)")
    cube_axes.SetZTitle("Z (km)")
    cube_axes.GetTitleTextProperty(0).SetColor(1, 1, 1)
    cube_axes.GetLabelTextProperty(0).SetColor(1, 1, 1)
    cube_axes.GetTitleTextProperty(1).SetColor(1, 1, 1)
    cube_axes.GetLabelTextProperty(1).SetColor(1, 1, 1)
    cube_axes.GetTitleTextProperty(2).SetColor(1, 1, 1)
    cube_axes.GetLabelTextProperty(2).SetColor(1, 1, 1)
    cube_axes.GetProperty().SetColor(0.8, 0.8, 0.8)
    viewer.add_actor(cube_axes)

    print("Launching PyViz4D Earth Viewer (Pure VTK)...")
    viewer.start()

if __name__ == '__main__':
    main()
