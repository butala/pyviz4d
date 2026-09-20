import argparse
import vtk
import pooch
from pyviz4d.viz import Viewer4D
from pyviz4d.cityjson import read_cityjson

def main():
    parser = argparse.ArgumentParser(description="PyViz4D CityJSON Demo")
    args = parser.parse_args()

    # Download a realistic CityJSON file (Rotterdam Railway LoD3 model)
    url = "https://3d.bk.tudelft.nl/opendata/cityjson/3dcities/v2.0/LoD3_Railway.city.json"
    print(f"Downloading real-world CityJSON sample from {url}...")
    filepath = pooch.retrieve(
        url=url,
        known_hash=None
    )

    print(f"Reading CityJSON from {filepath}...")
    # Read CityJSON
    polydata = read_cityjson(filepath)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    # Give the buildings a nice material property
    actor.GetProperty().SetColor(0.85, 0.85, 0.9)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetEdgeColor(0.3, 0.3, 0.3)
    actor.GetProperty().SetLineWidth(1.0)

    viewer = Viewer4D(bg_color=(0.15, 0.2, 0.25))
    viewer.add_actor(actor)

    viewer.ren.ResetCamera()

    print("Launching PyViz4D CityJSON Viewer...")
    viewer.start()

if __name__ == '__main__':
    main()
