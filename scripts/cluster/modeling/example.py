"""
Modeling: Large Scale
=====================

This script fits a `PointDict` dataset of a 'cluster-scale' strong lens with multiple lens galaxies where:

 - The cluster consists of a brightest cluster galaxy (BCG) which is modeled individually using an `EllIsothermal`
 profile.
 - The cluster contains a large scale dark matter halo which is modeled individually using an `SphNFWMCRLudlow`
 profile.
 - The cluster contains ~70 member galaxies, whose masses are modeled using `SphIsothermalSR` profiles whereby the mass
  of each galaxy is set via a `MassLightRelation`.
 - There are three observed source `Galaxy`'s which are modeled as a point source `PointSrcChi`.

The point-source dataset used in this example consists of the the positions of three lensed source's multiple images.

__This Example__

This script models an example strong lens on the 'cluster' end of the scale, where there is a large BCG, dark matter
halo and of order member galaxies. There are three lensed source galaxies that observed and used to fit the lens model.

In this example we model the sources as a point-sources, as fitting the full `Imaging` data and extended emission in
the lensed source's arcs is challenging due to the high complexity of the lens model.

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
__Downloading Data__

The **PyAutoLens** example cluster datasets are too large in filesize to distribute with the autolens workspace.

They are therefore stored in a separate GitHub repo:

 https://github.com/Jammy2211/autolens_cluster_datasets

Before running this script, make sure you have downloaded the example datasets and moved them to the folder 
`autolens_workspace/dataset/clusters`.

__Dataset__

Load the strong lens dataset `cluster`, which is the dataset we will use to perform lens modeling.

We begin by loading an image of the dataset. Although we perform point-source modeling and will not use this data in 
the model-fit, it is useful to load it for visualization. By passing this dataset to the model-fit at the
end of the script it will be used when visualizing the results. However, the use of an image in this way is entirely
optional, and if it were not included in the model-fit visualization would simple be performed using grids without
the image.
"""
dataset_name = "cluster"
dataset_path = path.join("dataset", "cluster", dataset_name)

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "f160w_image.fits"), pixel_scales=0.03
)

"""
__Point Source Dict__

We now load the point source dataset we will fit using point source modeling. We load this data as a `PointDict`,
which is a Python dictionary containing the positions of every source galaxy which is modeled as a point source. 

In this example there are three galaxies, whose multiple images are modeled as point sources corresponding to the 
brightest pixel of each lensed multiple image. 
"""
point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

"""
We can print this dictionary to see the `name` and `positions` of the dataset, as well as their noise-map values.
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

In other point-source modeling examples, we defined a `PositionsSolver`, which determines the multiple-images of 
a mass model for a point source at location (y,x) in the source plane, by iteratively ray-tracing light rays to the 
source-plane. 

For this example, a `PositionsSolver` is not used. This is because our model of every source galaxy uses 
the `PointSrcChi` model, which means the goodness-of-fit is evaluated in the source-plane. This removes the need to 
iteratively solve the lens equation. However, it is still good practise to define a `PositionsSolver` in a cluster
script, as we may wish to also perform image-plane fits.

Checkout the script ? for a complete description of this object, we will use the default `PositionSolver` in this 
example with a `point_scale_precision` half the value of the position noise-map, which should be sufficiently good 
enough precision to fit the lens model accurately.
"""
grid = al.Grid2D.uniform(
    shape_native=image.shape_native, pixel_scales=image.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.015)

"""
__Model__

We compose our lens model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a lens model where:

 - There is a brightest cluster galaxy (BCG) which is individually modeled using the `EllIsothermal` total mass 
 distribution [5 parameters].
 - There is a large scale dark matter halo component for the whole cluster, which is modeled using the `SphNFWMCRLudlow`
 profile [3 parameters].
 - The ~70 member galaxies are modeled collectively using the `MassLightRelation` scaling relation [2 parameters].
 - There are three source galaxy's whose light is a point `PointSrcChi` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=16.

__Model JSON File_

For cluster modeling, there can be many lens and source galaxies. Manually writing the model in a Python script, in the
way we do for galaxy-scale lenses, is therefore not feasible.
 
For this cluster, we therefore composed the the model by interfacing with Source Extractor 
(https://sextractor.readthedocs.io/) catalogue files. A full illustration of how to make the lens and source models 
from catalogue files is given in the following scripts:

 `autolens_workspace/notebooks/clusters/preprocess/tutorial_1_lens_model.ipynb` 
 `autolens_workspace/notebooks/clusters/preprocess/tutorial_2_sources.ipynb`  

This file is used to load the model below and it can be easily altered to compose a cluster model suited to your lens 
dataset!
"""
model_path = path.join("scripts", "cluster", "models", dataset_name)

lenses_file = path.join(model_path, "lenses.json")
lenses = af.Collection.from_json(file=lenses_file)

sources_file = path.join(model_path, "sources.json")
sources = af.Collection.from_json(file=sources_file)

galaxies = lenses + sources

model = af.Collection(galaxies=galaxies)

"""
__Name Pairing__

Every point-source dataset in the `PointDict` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` **PyAutoLens** will raise an error.
"""
print(model)

"""
__Search__

The lens model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/).

The folder `autolens_workspace/notebooks/imaging/modeling/customize/non_linear_searches` gives an overview of the 
non-linear searches **PyAutoLens** supports. If you are unclear of what a non-linear search is, checkout chapter 2 of 
the **HowToLens** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/clusters/cluster/large_scale/unique_identifier`.

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
    path_prefix=path.join("cluster"),
    name="large_scale",
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
