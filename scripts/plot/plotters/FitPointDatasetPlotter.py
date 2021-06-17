"""
Plots: PointDatasetPlotter
==========================

This example illustrates how to plot a `PointDataset` dataset using an `PointDatasetPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load an example strong lens `PointDataset` object.
"""
dataset_name = "mass_sie__source_point__0"
dataset_path = path.join("dataset", "point_source", dataset_name)

point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

point_dataset = point_dict["point_0"]

"""
__Fit__

We now fit it with a `Tracer` to create a `FitPointDataset` object.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(redshift=1.0, point_0=al.ps.Point(centre=(0.0, 0.0)))

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.025)

fit = al.FitPointDataset(
    point_dataset=point_dataset, tracer=tracer, positions_solver=positions_solver
)

"""
__Figures__

We now pass the FitPointDataset to a `FitPointDatasetPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
fit_imaging_plotter = aplt.FitPointDatasetPlotter(fit=fit)
fit_imaging_plotter.figures_2d(positions=True, fluxes=True)

"""
Finish.
"""
