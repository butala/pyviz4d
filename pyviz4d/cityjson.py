import json
import numpy as np
import vtk

def read_cityjson(filepath: str, target_epsg: int = None) -> vtk.vtkPolyData:
    """
    Reads a CityJSON file and converts it to a vtkPolyData object.
    Optionally transforms the coordinates to a target EPSG using pyproj.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vertices = np.array(data.get("vertices", []), dtype=np.float64)
    if "transform" in data:
        scale = data["transform"]["scale"]
        translate = data["transform"]["translate"]
        vertices = vertices * scale + translate

    # Coordinate transformation
    if target_epsg is not None and "metadata" in data and "referenceSystem" in data["metadata"]:
        try:
            from pyproj import CRS, Transformer
            source_crs_str = data["metadata"]["referenceSystem"]
            # Extract EPSG if possible, usually formatted like 'urn:ogc:def:crs:EPSG::7415'
            source_crs = CRS.from_user_input(source_crs_str)
            target_crs = CRS.from_epsg(target_epsg)
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            # Apply transformation
            xx, yy, zz = transformer.transform(
                vertices[:, 0], vertices[:, 1], vertices[:, 2]
            )
            vertices[:, 0] = xx
            vertices[:, 1] = yy
            vertices[:, 2] = zz
        except ImportError:
            print("Warning: pyproj is not installed, skipping coordinate transformation.")

    points = vtk.vtkPoints()
    for v in vertices:
        points.InsertNextPoint(v[0], v[1], v[2])

    polys = vtk.vtkCellArray()

    def process_boundary(boundary):
        if not boundary:
            return
        outer_ring = boundary[0]
        polys.InsertNextCell(len(outer_ring))
        for idx in outer_ring:
            polys.InsertCellPoint(idx)

    for obj_id, obj in data.get("CityObjects", {}).items():
        for geom in obj.get("geometry", []):
            geom_type = geom.get("type", "")
            boundaries = geom.get("boundaries", [])

            if geom_type == "Solid":
                for shell in boundaries:
                    for surface in shell:
                        process_boundary(surface)
            elif geom_type in ["MultiSurface", "CompositeSurface"]:
                for surface in boundaries:
                    process_boundary(surface)
            elif geom_type == "Surface":
                process_boundary(boundaries)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    return polydata
