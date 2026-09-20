import argparse
import numpy as np
import vtk
from pyviz4d.viz import Viewer4D, TemporalActor

class OrbitingPlanet(TemporalActor):
    def __init__(self, color, radius, orbit_distance, speed):
        self.orbit_distance = orbit_distance
        self.speed = speed

        source = vtk.vtkSphereSource()
        source.SetRadius(radius)
        source.SetThetaResolution(30)
        source.SetPhiResolution(30)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetSpecular(0.4)

        super().__init__(actor)

    def update(self, current_time: float):
        angle = current_time * self.speed
        x = self.orbit_distance * np.cos(angle)
        z = self.orbit_distance * np.sin(angle)
        self.actor.SetPosition(x, 0, z)

def main():
    parser = argparse.ArgumentParser(description="PyViz4D Grid Views Demo")
    parser.add_argument("--nrows", type=int, default=1, help="Number of rows")
    parser.add_argument("--ncols", type=int, default=2, help="Number of columns")
    args = parser.parse_args()

    print(f"Creating a {args.nrows}x{args.ncols} grid layout...")

    # 1. Initialize Viewer4D with the grid
    viewer = Viewer4D(size=(1200, 600), bg_color=(0.1, 0.1, 0.15), nrows=args.nrows, ncols=args.ncols)

    # 2. Common Object: The Central Sun
    sun_source = vtk.vtkSphereSource()
    sun_source.SetRadius(2.0)
    sun_source.SetThetaResolution(50)
    sun_source.SetPhiResolution(50)
    sun_mapper = vtk.vtkPolyDataMapper()
    sun_mapper.SetInputConnection(sun_source.GetOutputPort())
    sun_actor = vtk.vtkActor()
    sun_actor.SetMapper(sun_mapper)
    sun_actor.GetProperty().SetColor(1.0, 0.8, 0.1) # Yellow-Orange Sun
    # Add to ALL views by setting view=None
    viewer.add_actor(sun_actor, view=None)

    # 3. Common Object: A bounding box to show the shared 3D space
    cube = vtk.vtkCubeSource()
    cube.SetBounds(-10, 10, -5, 5, -10, 10)
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputConnection(cube.GetOutputPort())
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    cube_actor.GetProperty().SetRepresentationToWireframe()
    cube_actor.GetProperty().SetColor(0.4, 0.4, 0.4)
    # Add to ALL views
    viewer.add_actor(cube_actor, view=None)

    # 4. View-specific Objects
    # We will add an Earth-like planet exclusively to View 0 (Left/Top)
    earth = OrbitingPlanet(color=(0.2, 0.5, 1.0), radius=0.8, orbit_distance=6.0, speed=0.05)
    viewer.add_actor(earth, view=0)

    # And a Mars-like planet exclusively to View 1 (Right/Bottom)
    if args.nrows * args.ncols > 1:
        mars = OrbitingPlanet(color=(1.0, 0.3, 0.2), radius=0.5, orbit_distance=8.0, speed=0.03)
        viewer.add_actor(mars, view=1)

    # 5. Position Camera (since they are linked, manipulating the first sets it for all)
    cam = viewer.ren.GetActiveCamera()
    cam.SetPosition(0, 15, 25)
    cam.SetFocalPoint(0, 0, 0)
    cam.SetViewUp(0, 1, 0)

    # Ensure all renderers adjust clipping range to camera
    for ren in viewer.renderers:
        ren.ResetCameraClippingRange()

    viewer.add_playback_ui(max_time=300)

    print("Launching PyViz4D Grid Demo...")
    viewer.start(timer_interval_ms=16)

if __name__ == '__main__':
    main()
