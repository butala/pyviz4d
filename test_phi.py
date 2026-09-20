from phi.flow import *
import numpy as np

res = 16
velocity = StaggeredGrid(0, extrapolation.ZERO, x=res, y=res*2, z=res, bounds=Box(x=(0,100), y=(0,200), z=(0,100)))
density = CenteredGrid(0, extrapolation.BOUNDARY, x=res, y=res*2, z=res, bounds=Box(x=(0,100), y=(0,200), z=(0,100)))

inflow = 0.2 * CenteredGrid(Sphere(x=50, y=10, z=50, radius=10), extrapolation.BOUNDARY, x=res, y=res*2, z=res, bounds=Box(x=(0,100), y=(0,200), z=(0,100)))

density = advect.mac_cormack(density, velocity, dt=1) + inflow
buoyancy = (density * vec(x=0, y=0.1, z=0)).at(velocity)
velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy
velocity, _ = fluid.make_incompressible(velocity)

arr = density.values.numpy('x,y,z')
print("Shape:", arr.shape)
