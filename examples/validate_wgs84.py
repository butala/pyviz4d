import vtk
from pyviz4d.viz import EarthViewer4D
from pyviz4d.earth import WGS84, wgs84_to_cartesian, sphere_to_cartesian

def create_marker(x, y, z, color, radius=50.0):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetThetaResolution(20)
    source.SetPhiResolution(20)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.SetPosition(x, y, z)
    return actor

def create_wireframe_sphere(radius):
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetThetaResolution(50)
    source.SetPhiResolution(50)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToWireframe()
    actor.GetProperty().SetColor(0.0, 1.0, 1.0) # Cyan
    actor.GetProperty().SetOpacity(0.3)
    return actor

def main():
    viewer = EarthViewer4D()

    a_km = WGS84.a / 1e3

    # Let's pick a high-latitude location where the bulge discrepancy is severe.
    # We will use the geographic North Pole (Lat 90, Lon 0) to maximize the difference.
    test_lat = 90.0
    test_lon = 0.0

    # 1. Calculate and plot the WGS84 accurate coordinate (Green)
    x_wgs, y_wgs, z_wgs = wgs84_to_cartesian(test_lat, test_lon)
    wgs_marker = create_marker(x_wgs, y_wgs, z_wgs, color=(0.0, 1.0, 0.0), radius=30)
    viewer.add_actor(wgs_marker)

    # 2. Calculate and plot assuming a perfect sphere using the equatorial radius (Red)
    x_sph, y_sph, z_sph = sphere_to_cartesian(test_lat, test_lon, radius_km=a_km)
    sph_marker = create_marker(x_sph, y_sph, z_sph, color=(1.0, 0.0, 0.0), radius=30)
    viewer.add_actor(sph_marker)

    # 3. Add a transparent wireframe sphere of radius 'a' to visualize the bulge
    wireframe = create_wireframe_sphere(a_km)
    viewer.add_actor(wireframe)

    # 4. Add scientific axes
    cube_axes = vtk.vtkCubeAxesActor()
    cube_axes.SetBounds(-7000, 7000, -7000, 7000, -7000, 7000)
    cube_axes.SetCamera(viewer.ren.GetActiveCamera())
    cube_axes.GetProperty().SetColor(0.8, 0.8, 0.8)
    for i in range(3):
        cube_axes.GetTitleTextProperty(i).SetColor(1, 1, 1)
        cube_axes.GetLabelTextProperty(i).SetColor(1, 1, 1)
    viewer.add_actor(cube_axes)

    print("==================================================")
    print("WGS84 Validation Demo")
    print(f"Testing Lat: {test_lat}, Lon: {test_lon}")
    print(f"True WGS84 Z-coordinate:   {z_wgs:.2f} km (Green Marker)")
    print(f"Spherical Z-coordinate:    {z_sph:.2f} km (Red Marker)")
    print(f"Discrepancy (Error):       {abs(z_sph - z_wgs):.2f} km")
    print("==================================================")

    # Move camera to the North Pole to see the discrepancy clearly
    viewer.ren.GetActiveCamera().SetPosition(0, 4000, a_km + 1500)
    viewer.ren.GetActiveCamera().SetFocalPoint(0, 0, a_km)
    viewer.ren.GetActiveCamera().SetViewUp(0, 1, 0)
    viewer.ren.ResetCameraClippingRange()

    viewer.start()

if __name__ == '__main__':
    main()
