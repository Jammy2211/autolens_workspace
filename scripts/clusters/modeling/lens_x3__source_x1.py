"""
Modeling: Point-Source Mass Total
=================================

In this script, we fit a `PointDict` of a strong lens with multiple lens galaxies where:

 - The cluster consists of three whose light models are `SphSersic` profiles and total mass distributions
 are `SphIsothermal` models.
 - The source `Galaxy` is modeled as a point source `Point`.

The point-source dataset used in this example consists of the the positions of the lensed source's multiple images and
their fluxes, both of which are used in the fit.

__Cluster Scale__

Strongly lensed galaxy clusters vary in scale from two extremes:

 - Groups: Objects with a 'primary' lens galaxy and a handful of lower mass galaxies nearby. They typically have just
 one or two lensed sources whose arcs and extended emission are visible, with Einstein Radii in the range 5.0" -> 10.0".

 - Clusters: Objects with tens or hundreds of lens galaxies and lensed sources.

**PyAutoLens** has tools for modeling any dataset between these two extremes, and the example datasets available
throughout the `autolens_workspace` illustrate the different tools that be used for modeling clusters of different
scales.

__This Example__

This script models an example galaxy on the 'group' end of the scale, where there is a single primary lens galaxy
and two smaller galaxies nearby, whose mass contributes significantly to the ray-tracing and is therefore included in
the strong lens model.

In this example we model the source as a point-source, as fitting the full `Imaging` data and extended emission in the
lensed source's arcs is challenging due to the high complexity of the lens model.

The `clusters/chaining` package includes an example script showing how **PyAutoLens** can model this dataset's full
extended emission, however this requires familiarity with **PyAutoLens**'s advanced feature called 'search chaining'
which is covered in chapter 3 of **HowToLens**. This package also shows how to do this using a pixelized source
reconstruction.
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

Load the strong lens dataset `lens_x3__source_x1`, which is the dataset we will use to perform lens modeling.

We begin by loading an image of the dataset. Although we perform point-source modeling and will not use this data in 
the model-fit, it is useful to load it for visualization. By passing this dataset to the model-fit at the
end of the script it will be used when visualizing the results. However, the use of an image in this way is entirely
optional, and if it were not included in the model-fit visualization would simple be performed using grids without
the image.
"""
dataset_name = "lens_x3__source_x1"
dataset_path = path.join("dataset", "clusters", dataset_name)

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"), pixel_scales=0.1
)

"""
We now load the point source dataset we will fit using point source modeling. We load this data as a `PointDict`,
which is a Python dictionary containing the positions and fluxes of every point source. 

In this example there is just one point source, corresponding to the bright pixel of each lensed multiple image. For
other galaxy cluster examples point source datasets and models containing multiple sources are fitted.
"""
point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

"""
We can print this dictionary to see the `name`, `positions` and `fluxes` of the dataset, as well as their noise-map 
values.
"""
print("Point Source Dict:")
print(point_dict)

"""
We can plot our positions dataset over the observed image.
"""
visuals_2d = aplt.Visuals2D(positions=point_dict.positions_list)

array_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
array_plotter.figure_2d()

"""
We can also just plot the positions, omitting the image.
"""
grid_plotter = aplt.Grid2DPlotter(grid=point_dict["point_0"].positions)
grid_plotter.figure_2d()

"""
__PositionsSolver__

For point-source modeling we also need to define our `PositionsSolver`. This object determines the multiple-images of 
a mass model for a point source at location (y,x) in the source plane, by iteratively ray-tracing light rays to the 
source-plane. 

Checkout the script ? for a complete description of this object, we will use the default `PositionSolver` in this 
example with a `point_scale_precision` half the value of the position noise-map, which should be sufficiently good 
enough precision to fit the lens model accurately.
"""
grid = al.Grid2D.uniform(
    shape_native=image.shape_native, pixel_scales=image.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.025)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - There are three lens galaxy's with `SphIsothermal` total mass distributions, with the prior on the centre of each 
 profile informed by its observed centre of light [9 parameters].
 - The source galaxy's light is a point `PointFlux` [3 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.

__Model JSON File_

For cluster modeling, there can be many lens and source galaxies. Manually writing the model in a Python script, in the
way we do for galaxy-scale lenses, is therefore not ideal.

We therefore write the model for this system in a separate Python file and output it to a .json file, which we created 
via the script `clusters/modeling/models/lens_x3__source_x1.py` and can be found in the 
file `clusters/modeling/models/lens_x3__source_x1.json`. 

This file is used to load the model below and it can be easily altered to compose a cluster model suited to your lens 
dataset!

__Name Pairing__

Every point-source dataset in the `PointDict` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` **PyAutoLens** will raise an error.

In this example, where there is just one source, name pairing appears unecessary. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
fitted to its particular lensed images in the `PointDict`!
**PyAutoLens** assumes that the lens galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the  lens is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autolens_workspace/notebooks/preprocess`). 
 - Manually override the lens model priors (`autolens_workspace/notebooks/imaging/modeling/customize/priors.py`).
"""
model_path = path.join("scripts", "clusters", "modeling", "models")
model_file = path.join(model_path, "lens_x3__source_x1.json")

model = af.Collection.from_json(file=model_file)

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/clusters/mass_sie__source_sersic/mass[sie]_source[point]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.

An identical combination of model, search and dataset generates the same identifier, meaning that rerunning the
script will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset,
a new unique identifier will be generated, ensuring that the model-fit results are output into a separate folder. 

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Dynesty uses parallel processing to sample multiple 
lens models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core process you should be able to use `number_of_cores=4`. For 
users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.
"""
search = af.DynestyStatic(
    path_prefix=path.join("clusters"),
    name="mass[sie]_source[point]",
    unique_tag=dataset_name,
    nlive=50,
    number_of_cores=1,
)

"""
__Analysis__

The `AnalysisPoint` object defines the `log_likelihood_function` used by the non-linear search to fit the model 
to the `PointDataset`.
"""
analysis = al.AnalysisPoint(point_dict=point_dict, solver=positions_solver)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The lens model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Tracer` object.
 - Information on the posterior as estimated by the `Dynesty` non-linear search.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grid
)
tracer_plotter.subplot_tracer()

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

"""
Checkout `autolens_workspace/notebooks/modeling/results.py` for a full description of the result object.
"""
