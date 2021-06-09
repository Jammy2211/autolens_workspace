"""
Plots: MassProfilePlotter
=========================

This example illustrates how to plot a `MassProfile` using a `MassProfilePlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
__Mass Profile__

First, lets create a simple `MassProfile` which we'll plot.
"""
mass = al.mp.EllIsothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, angle=45.0),
)

"""
__Grid__

We also need the 2D grid the `MassProfile` is evaluated on.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Figures__

We now pass the mass profile and grid to a `MassProfilePlotter` and call various `figure_*` methods to 
plot different attributes in 1D and 2D.
"""
mass_profile_plotter = aplt.MassProfilePlotter(mass_profile=mass, grid=grid)
mass_profile_plotter.figures_2d(
    convergence=True,
    potential=True,
    deflections_y=True,
    deflections_x=True,
    magnification=True,
)
mass_profile_plotter.figures_1d(convergence=True, potential=True)

"""
__Include__

A `MassProfile` and its `Grid2D` contains the following attributes which can be plotted automatically via 
the `Include2D` object.

(By default, a `Grid2D` does not contain a `Mask2D`, we therefore manually created a `Grid2D` with a mask to illustrate
plotting its mask and border below).
"""
include_2d = aplt.Include2D(
    origin=True, mask=True, border=True, mass_profile_centres=True, critical_curves=True
)

mask = al.Mask2D.circular_annular(
    shape_native=grid.shape_native,
    pixel_scales=grid.pixel_scales,
    inner_radius=0.3,
    outer_radius=2.0,
)
masked_grid = al.Grid2D.from_mask(mask=mask)

mass_profile_plotter = aplt.MassProfilePlotter(
    mass_profile=mass, grid=masked_grid, include_2d=include_2d
)
mass_profile_plotter.figures_2d(convergence=True)

"""
Finish.
"""
