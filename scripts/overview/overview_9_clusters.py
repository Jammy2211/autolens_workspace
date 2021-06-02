"""
Overview: Clusters
------------------

Galaxy clusters are the beasts of strong lensing. They contain tens or hundreds of lens galaxies and lensed sources,
with lensed sources at many different redshifts requiring full multi-plane ray-tracing calculations. They contain one
or more brightest cluster galaxy(s) a large scale dark matter halo and have arcs with Einstein Radii 10.0" -> 100.0"
and beyond.

A cluster scale strong lens model is typically composed of the following:

 - One or more brightest cluster galaxies (BCG), which are sufficiently large that we model them individually.

 - One or more cluster-scale dark matter halos, which are again modeled individually.

 - Tens or hundreds of galaxy cluster member galaxies. The low individual masses of these objects means we cannot
  model them individually are constrain their mass, but their collectively large enough mass to need modeling. These
  are modeled using a scaling relation which assumes that light traces mass, where the luminosity of each individual
  galaxy is used to set up this scaling relation.

 - Tens or hundreds of source galaxies, each with multiple sets of images that constrain the lens model. These are
 modeled as a point-source, although **PyAutoLens** includes tools for modeling the imaging data of sources once a good
 lens model is inferred. The source redshifts are also used to account for multi-plane ray-tracing.

This overview uses **PyAutoLens** to model the example strong lens cluster SDSS1152P3312, with individual models for
cluster's BCG and dark matter halo . The cummulative lensing of 70 member galaxies is included via a scaling relation
that assumes mass traces light and three background sources are observed and modeled as point-sources.
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

Lets plot the image of SDSS1152P3312.
"""
dataset_name = "sdssj1152p3312"
dataset_path = path.join("dataset", "cluster")

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "f160w_image.fits"), hdu=0, pixel_scales=0.03
)

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=0.1))

array_plotter = aplt.Array2DPlotter(array=image.native, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

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
__PositionsSolver__

In other point-source modeling examples, we defined a `PositionsSolver`, which determines the multiple-images of 
a mass model for a point source at location (y,x) in the source plane, by iteratively ray-tracing light rays to the 
source-plane. 

For this example, a `PositionsSolver` is not used. This is because our model of every source galaxy uses 
the `PointSrcChi` model, which means the goodness-of-fit is evaluated in the source-plane. This removes the need to 
iteratively solve the lens equation. However, it is still good practise to define a `PositionsSolver` in a cluster
script, as we may wish to also perform image-plane fits.
"""
grid = al.Grid2D.uniform(
    shape_native=image.shape_native, pixel_scales=image.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.015)

"""
__Model_

Composing the lens model for cluster scale objects requires care, given there are could be hundreds of lenses and 
sources galaxies. Manually writing the model in a Python script, in the way we do for galaxy-scale lenses, is therefore 
not feasible.

For this cluster, we therefore composed the the model by interfacing with Source Extractor 
(https://sextractor.readthedocs.io/) catalogue files. A full illustration of how to make the lens and source models 
from catalogue files is given in the following scripts:

 `autolens_workspace/notebooks/cluster/model_maker/example__lenses.ipynb` 
 `autolens_workspace/notebooks/cluster/model_maker/example__sources.ipynb`  

These files are used to load the model below and they can be easily altered to compose a cluster model suited to your lens 
dataset!

For this cluster model, we set up every source galaxy as a `PointSrcChi` model. This evaluates 
a 'source-plane chi-squared', that is computationally much faster to evaluate than the image-plane chi-squared user in
other point-source examples. This is a common assumption made in cluster lens models, of course **PyAutoLens**
fully supports cluster models using an image-plane chi-squared.
"""
model_path = path.join("scripts", "cluster", "models", dataset_name)

lenses_file = path.join(model_path, "lenses.json")
lenses = af.Collection.from_json(file=lenses_file)

sources_file = path.join(model_path, "sources.json")
sources = af.Collection.from_json(file=sources_file)

galaxies = lenses + sources

model = af.Collection(galaxies=galaxies)

"""
If we print the `Model` we can see that it contains ALOT of galaxies.
"""
print(model)

"""
__Search + Analysis + Model-Fit__

We are now able to model this dataset as a point source, using the exact same tools we used in the point source 
overview.
"""
search = af.DynestyStatic(name="overview_clusters")

analysis = al.AnalysisPoint(point_dict=point_dict, solver=positions_solver)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result contains information on the BCG, cluster scale dark matter halo and mass-light scaling relation:
"""
print(result.max_log_likelihood_instance.galaxies.bcg.mass)
print(result.max_log_likelihood_instance.galaxies.dark.mass)
print(result.max_log_likelihood_instance.galaxies.scaling_relation)

"""
__Extended Source Fitting__

For clsuter-scale lenses fitting the extended surface-brightness is extremely difficult. The models become high 
dimensional and difficult to fit, and it becomes very computationally. Furthermore, the complexity of cluster mass 
models can make it challenging to compose a mass model which is sufficiently accurate that a source reconstruction is
even feasible!

Nevertheless, we are currently developing tools that try and make this possible. These will take approaches like 
fitting individual sources after modeling the entire cluster as a point-source and parallelizing the model-fitting
process out in a way that 'breaks-up' the model-fitting procedure.

These tools are in-development, but we are keen to have users with real sciences cases trial them as we develop
them. If you are interested please contact me! (https://github.com/Jammy2211).

__Wrap Up__

The `cluster` package of the `autolens_workspace` contains numerous example scripts for performing cluster-sale modeling 
and simulating cluster-scale strong lens datasets.
"""
