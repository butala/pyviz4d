import numpy as np
import vtk
from collections import namedtuple

from .config import CACHE_PATH
from .blue_marble import fetch

class Ellipsoid(namedtuple('Ellipsoid', 'a f_inv')):
    """
    a:     semi-major axis [m]
    f_inv: flattening factor inverse
    """
    @property
    def f(self):
        return 1.0 / self.f_inv

    @property
    def b(self):
        return self.a * (1.0 - self.f)

WGS84 = Ellipsoid(6378137.0, 298.257223563)

def wgs84_to_cartesian(lat_deg, lon_deg, alt_km=0.0):
    """
    Converts Geodetic (WGS84) latitude, longitude, and altitude to
    3D Cartesian (X, Y, Z) coordinates in km.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)

    a = WGS84.a / 1e3
    e2 = 2 * WGS84.f - WGS84.f**2

    # Prime vertical radius of curvature
    N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

    x = (N + alt_km) * np.cos(lat) * np.cos(lon)
    y = (N + alt_km) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt_km) * np.sin(lat)

    return x, y, z

def sphere_to_cartesian(lat_deg, lon_deg, radius_km, alt_km=0.0):
    """
    Converts spherical latitude, longitude, and altitude to
    3D Cartesian coordinates, assuming a perfect sphere.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = radius_km + alt_km

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return x, y, z

def earth_actor(theta_resolution=100, phi_resolution=100, resolution='low', cache_path=CACHE_PATH):
    """
    Returns a pure VTK actor of the textured Earth ellipsoid.
    """
    tex_path = fetch(cache_path, resolution=resolution)

    a_km = WGS84.a / 1e3
    b_km = WGS84.b / 1e3

    # 1. Create standard textured sphere
    source = vtk.vtkTexturedSphereSource()
    source.SetRadius(a_km)
    source.SetThetaResolution(theta_resolution)
    source.SetPhiResolution(phi_resolution)

    # 2. Read texture (since we download .jpg in blue_marble.py)
    reader = vtk.vtkJPEGReader()
    reader.SetFileName(tex_path)

    texture = vtk.vtkTexture()
    texture.SetInputConnection(reader.GetOutputPort())

    # 3. Transform Z to make it an ellipsoid matching WGS84
    # We also rotate 180 degrees around Z to align the visual Blue Marble texture
    # with the ECEF math (+X = Prime Meridian, +Y = 90 deg East, -Y = North America)
    transform = vtk.vtkTransform()
    transform.RotateZ(180)
    transform.Scale(1.0, 1.0, b_km / a_km)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(source.GetOutputPort())
    transform_filter.SetTransform(transform)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.SetTexture(texture)

    # Improve lighting by boosting ambient and diffuse properties
    # This prevents harsh, deep shadows on the dark side of the globe
    actor.GetProperty().SetAmbient(0.4)
    actor.GetProperty().SetDiffuse(0.8)
    actor.GetProperty().SetSpecular(0.1)

    return actor
