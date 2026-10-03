from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import vtk

from .volume import create_vtk_image_from_numpy


class VTKSeriesWriter:
    """
    Exports time-series components to binary VTK XML files (.vti, .vtp)
    and maintains a .pvd (ParaView Data) collection manifest.
    """
    def __init__(self, output_dir: str, collection_name: str = "series"):
        self.output_dir = output_dir
        self.collection_name = collection_name
        self.pvd_path = os.path.join(output_dir, f"{collection_name}.pvd")
        self.entries = []  # tuples of (timestep, relative_filename)
        os.makedirs(self.output_dir, exist_ok=True)
        # Built once and appended to.  Rebuilding the manifest from
        # self.entries on every frame is O(n^2) over a series; this keeps each
        # write O(1) while still leaving a valid .pvd on disk at all times.
        self._root = ET.Element("VTKFile", type="Collection", version="0.1",
                                byte_order="LittleEndian")
        self._collection = ET.SubElement(self._root, "Collection")

    def _record(self, timestep: float, filename: str):
        """Register a written frame and refresh the .pvd manifest."""
        self.entries.append((timestep, filename))
        ET.SubElement(self._collection, "DataSet", timestep=str(timestep),
                      group="", part="0", file=filename)
        self._write_pvd()

    def write_image_data(self, data: np.ndarray | vtk.vtkImageData, timestep: float,
                         filename_prefix: str = "frame", spacing=(1.0, 1.0, 1.0)):
        """
        Writes a 3D numpy array or vtkImageData as a binary .vti file and records the timestep.
        """
        if isinstance(data, np.ndarray):
            vtk_image = create_vtk_image_from_numpy(data, spacing=spacing)
        elif isinstance(data, vtk.vtkImageData):
            vtk_image = data
        else:
            raise TypeError("Data must be a 3D numpy array or vtkImageData")

        filename = f"{filename_prefix}_{len(self.entries):05d}.vti"
        full_path = os.path.join(self.output_dir, filename)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(full_path)
        writer.SetInputData(vtk_image)
        writer.SetDataModeToBinary()
        writer.Write()

        self._record(timestep, filename)
        return full_path

    def write_poly_data(self, poly_data: vtk.vtkPolyData, timestep: float,
                        filename_prefix: str = "mesh"):
        """
        Writes a vtkPolyData object as a binary .vtp file and records the timestep.
        """
        filename = f"{filename_prefix}_{len(self.entries):05d}.vtp"
        full_path = os.path.join(self.output_dir, filename)

        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(full_path)
        writer.SetInputData(poly_data)
        writer.SetDataModeToBinary()
        writer.Write()

        self._record(timestep, filename)
        return full_path

    def _write_pvd(self):
        """Writes the .pvd XML manifest."""
        tree = ET.ElementTree(self._root)
        ET.indent(tree, space="  ", level=0)
        tree.write(self.pvd_path, xml_declaration=True, encoding="utf-8")


def parse_pvd(pvd_path: str) -> list[tuple]:
    """
    Parses a .pvd file into a sorted list of (timestep, absolute_filepath).
    """
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    base_dir = os.path.dirname(os.path.abspath(pvd_path))
    collection = root.find("Collection")
    if collection is None:
        return []

    entries = []
    for dataset in collection.findall("DataSet"):
        ts = float(dataset.get("timestep", 0.0))
        rel_file = dataset.get("file")
        abs_file = os.path.join(base_dir, rel_file)
        entries.append((ts, abs_file))

    entries.sort(key=lambda x: x[0])
    return entries
