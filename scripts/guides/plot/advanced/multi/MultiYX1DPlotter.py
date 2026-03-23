"""
Plots: Multi 1D Profile Plots
==============================

This example shows how to plot multiple 1D profiles on the same figure.

The specific example plots the 1D convergence profiles of two different `MassProfile` objects
side-by-side on a single matplotlib figure.

In the old API, this was done using a `MultiYX1DPlotter` object. That class no longer exists.

In the new API, 1D profiles are computed directly and plotted using standard matplotlib.

__Start Here Notebook__

If any code in this script is unclear, refer to `plot/start_here.ipynb`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import autolens as al
import autolens.plot as aplt

"""
__Mass Profiles__

Create two mass profiles whose 1D convergences we will plot on the same figure.
"""
mass_0 = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.0,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
)

mass_1 = al.mp.PowerLaw(
    centre=(0.0, 0.0),
    einstein_radius=1.0,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    slope=2.1,
)

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__1D Convergence Profiles__

Compute the 1D convergence profile for each mass profile.

We use a projected 2D radial grid centred on the profile and aligned with its major axis.
Each mass profile's `convergence_2d_from()` is called with this grid to get the 1D profile.
"""
grid_2d_projected_0 = grid.grid_2d_radial_projected_from(
    centre=mass_0.centre, angle=mass_0.angle
)
convergence_1d_0 = mass_0.convergence_2d_from(grid=grid_2d_projected_0)

grid_2d_projected_1 = grid.grid_2d_radial_projected_from(
    centre=mass_1.centre, angle=mass_1.angle
)
convergence_1d_1 = mass_1.convergence_2d_from(grid=grid_2d_projected_1)

"""
__Plot__

Plot both profiles on the same matplotlib figure.
"""
plt.figure(figsize=(8, 5))
plt.plot(grid_2d_projected_0[:, 1], convergence_1d_0, label="Isothermal")
plt.plot(grid_2d_projected_1[:, 1], convergence_1d_1, label="PowerLaw (slope=2.1)")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Convergence")
plt.title("1D Convergence Profiles")
plt.legend()
plt.show()
plt.close()

"""
__Log Scale__

Convergence profiles are often easier to interpret in log-log space.
"""
plt.figure(figsize=(8, 5))
plt.semilogy(grid_2d_projected_0[:, 1], convergence_1d_0, label="Isothermal")
plt.semilogy(grid_2d_projected_1[:, 1], convergence_1d_1, label="PowerLaw (slope=2.1)")
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Convergence")
plt.title("1D Convergence Profiles (Log Scale)")
plt.legend()
plt.show()
plt.close()

"""
__Mass Profile 2D Plots__

2D mass profile quantities are plotted with `aplt.plot_array()`.
"""
aplt.plot_array(array=mass_0.convergence_2d_from(grid=grid), title="Isothermal Convergence")
aplt.plot_array(array=mass_1.convergence_2d_from(grid=grid), title="PowerLaw Convergence")
