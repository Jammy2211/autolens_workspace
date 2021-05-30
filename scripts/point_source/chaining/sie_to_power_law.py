"""
Chaining: SIE to Power-law
==========================

This script chains two searches to fit `PointDict` data of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's total mass distribution is an `EllPowerLaw`.
 - The source `Galaxy` is a point source `Point`.

The two searches break down as follows:

 1) Models the lens galaxy's mass as an `EllIsothermal` and the source galaxy's as a point `Point`.

 2) Models the lens galaxy's mass an an `EllPowerLaw` and the source galaxy's as a point `Point`.

__Why Chain?__

The `EllPower` is a general form of the `EllIsothermal` which has one additional parameter, the `slope`,
which controls the inner mass distribution as follows:

 - A higher slope concentrates more mass in the central regions of the mass profile relative to the outskirts.
 - A lower slope shallows the inner mass distribution reducing its density relative to the outskirts. 

By allowing the lens model to vary the mass profile's inner distribution, its non-linear parameter space becomes
significantly more complex and a notable degeneracy appears between the mass model`s mass normalization, elliptical
components and slope. This is challenging to sample in an efficient and robust manner, especially when the non-linear
search's initial samples use broad uniform priors on the lens and source parameters.

Search chaining allows us to begin by fitting an `EllIsothermal` model and therefore estimate the lens's mass
model and the source parameters via a non-linear parameter space that does not have a strong of a parameter degeneracy
present. This makes the model-fit more efficient and reliable.

The second search then fits the `EllPowerLaw`, using prior passing to initialize the mass and elliptical
components of the lens galaxy as well as the source galaxy's point-source (y,x) centre.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__ 

Load and plot the `Imaging` of the point-source dataset, purely for visualization of the strong lens.
"""
dataset_name = "mass_sie__source_point__0"
dataset_path = path.join("dataset", "point_source", dataset_name)

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"), pixel_scales=0.05
)

"""
__PointDict__

Load and plot the `PointDict` dataset, which is the dataset used to perform lens modeling.
"""
point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

print("Point Source Multiple Image (y,x) Arc-second Coordinates:")
print(point_dict["point_0"].positions.in_list)

visuals_2d = aplt.Visuals2D(positions=point_dict.positions_list)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
array_plotter.figure_2d()

grid_plotter = aplt.Grid2DPlotter(grid=point_dict["point_0"].positions)
grid_plotter.figure_2d()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("point_source", "chaining", "sie_to_power_law")

"""
__PositionsSolver__

Setup the `PositionSolver`.
"""
grid = al.Grid2D.uniform(
    shape_native=image.shape_native, pixel_scales=image.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.025)

"""
__Model (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's total mass distribution is an `EllIsothermal` [5 parameters].
 - The source galaxy's is a point `Point` [2 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=9.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.EllIsothermal)
source = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.Point)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix, name="search[1]__sie", unique_tag=dataset_name, nlive=50
)

analysis = al.AnalysisPoint(point_dict=point_dict, solver=positions_solver)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - The lens galaxy's total mass distribution is an `EllPowerLaw` [6 parameters: priors initialized from search 1].
 - The source galaxy's light is again a point `Point` [2 parameters: priors initialized from search 1].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=10.

The term `model` below passes the source model as model-components that are to be fitted for by the  non-linear search. 
Because the source model does not change we can pass its priors by simply using the`model` attribute of the result:
"""
source = result_1.model.galaxies.source

"""
However, we cannot use this to pass the lens galaxy, because its mass model must change from an `EllIsothermal` 
to an `EllPowerLaw`. The following code would not change the mass model to an `EllPowerLaw`:
 
 `lens = result.model.galaxies.lens`
 
We can instead use the `take_attributes` method to pass the priors. Below, we pass the lens of the result above to a
new `EllPowerLaw`, which will find all parameters in the `EllIsothermal` model that share the same name
as parameters in the `EllPowerLaw` and pass their priors (in this case, the `centre`, `elliptical_comps` 
and `einstein_radius`).

This leaves the `slope` parameter of the `EllPowerLaw` with its default `UniformPrior` which has a 
`lower_limit=1.5` and `upper_limit=3.0`.
"""
mass = af.Model(al.mp.EllPowerLaw)
mass.take_attributes(result_1.model.galaxies.lens.mass)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search + Analysis + Model-Fit (Search 2)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the `model.info` file of the search 2 model-fit to ensure the priors were passed correctly, as 
well as the checkout the results to ensure an accurate power-law mass model is inferred.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]__power_law",
    unique_tag=dataset_name,
    nlive=75,
)

analysis = al.AnalysisPoint(point_dict=point_dict, solver=positions_solver)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a lens mass model as an `EllIsothermal` and 
passed its priors to then fit the more complex `EllPowerLaw` model. 

This removed difficult-to-fit degeneracies from the non-linear parameter space in search 1, providing a more robust 
and efficient model-fit.
"""
