from phi.flow import *

res = 24
bounds = Box(x=(0, 100), y=(0, 200), z=(0, 100))
velocity = StaggeredGrid(0, extrapolation.ZERO, x=res, y=res*2, z=res, bounds=bounds)
density = CenteredGrid(0, extrapolation.BOUNDARY, x=res, y=res*2, z=res, bounds=bounds)
source_sphere = Sphere(x=50, y=10, z=50, radius=10)
inflow = 0.2 * CenteredGrid(source_sphere, extrapolation.BOUNDARY, x=res, y=res*2, z=res, bounds=bounds)

density = advect.mac_cormack(density, velocity, dt=1) + inflow
buoyancy = (density * vec(x=0, y=0.1, z=0)).at(velocity)
velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy

try:
    # try relaxing tolerance
    velocity, _ = fluid.make_incompressible(velocity, solve=Solve('auto', rel_tol=1e-3, abs_tol=1e-3))
    print("Success with relaxed tolerance.")
except Exception as e:
    print(f"Error: {e}")
