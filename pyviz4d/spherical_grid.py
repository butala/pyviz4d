import vtk
import numpy as np

def spherical_grid_actor(r1, r2, N_theta, N_phi):
    """
    Generates a VTK Actor containing lines representing
    spokes in a spherical grid.
    """
    theta_vec = np.linspace(0, np.pi, N_theta + 1)[1:-1]
    phi_vec = np.linspace(0, 2 * np.pi, N_phi, endpoint=False) + np.pi / N_phi

    thetas, phis = np.meshgrid(theta_vec, phi_vec)
    thetas = np.concatenate(thetas)
    phis = np.concatenate(phis)

    r1s = np.full_like(thetas, r1)
    r2s = np.full_like(thetas, r2)

    x1s = r1s * np.sin(thetas) * np.cos(phis)
    y1s = r1s * np.sin(thetas) * np.sin(phis)
    z1s = r1s * np.cos(thetas)

    x2s = r2s * np.sin(thetas) * np.cos(phis)
    y2s = r2s * np.sin(thetas) * np.sin(phis)
    z2s = r2s * np.cos(thetas)

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()

    # Construct VTK lines connectivity
    for i, (x1, y1, z1, x2, y2, z2) in enumerate(zip(x1s, y1s, z1s, x2s, y2s, z2s)):
        p1_id = points.InsertNextPoint(x1, y1, z1)
        p2_id = points.InsertNextPoint(x2, y2, z2)

        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, p1_id)
        line.GetPointIds().SetId(1, p2_id)
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.2, 0.2, 0.2)
    actor.GetProperty().SetLineWidth(2.0)

    return actor
