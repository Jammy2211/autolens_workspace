"""
Chaining: Point Source to Imaging
=================================

In this script, we chain three searches to fit `Imaging` of a strong lens with multiple lens galaxies where:

 - The cluster consists of three whose light models are `SphSersic` profiles and total mass distributions
 are `SphIsothermal` models.
 - The source galaxy's `LightProfile` is an `EllSersic`.

The two searches break down as follows:

 1) Model the lens galaxy masses with a point source galaxy, fitting just the position information in the source.
 2) Model the full surface brightness information in the `Imaging` data using `LightProfile`'s for the lens galaxies
 and lensed source.

__Why Chain?__

There are a number of benefits of chaining a point source fit to an imaging fit, as opposed to doing just one fit:

 - The point source fit is lower dimensionality than a light profile fit and computationally very fast. It can
 therefore provide accurate estimates for the lens and source model parameters. However, the point source fit does
 not extract anywhere near the maximal amount of information from the data.

 - The fit to the imaging data is much higher dimensionality and computationally slower. It would be challenging to
 fit this model without accurate initialization of the lens model parameters. However, it extracts a lot more
 information from the data.

This script therefore initializes the lens model efficiently using a point-source fit and then switches to a full
fit on the imaging data.
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

Load the strong lens dataset `lens_x3__source_x1` point source dataset and imaging, and plot them.
"""
dataset_name = "lens_x3__source_x1"
dataset_path = path.join("dataset", "clusters", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

point_source_dict = al.PointSourceDict.from_json(
    file_path=path.join(dataset_path, "point_source_dict.json")
)

visuals_2d = aplt.Visuals2D(positions=point_source_dict.positions_list)

array_plotter = aplt.Array2DPlotter(array=imaging.image, visuals_2d=visuals_2d)
array_plotter.figure_2d()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("clusters", "chaining", "point_source_to_imaging")

"""
__PositionsSolver__

Define the position solver used for the point source fitting.
"""
grid = al.Grid2D.uniform(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.025)

"""
__Model (Search 1)__

Compose the lens model by loading it from a .json file made in the file `clusters/modeling/models/lens_x3__source_x1.py`:

 - There are three lens galaxy's with `SphIsothermal` total mass distributions, with the prior on the centre of each 
 profile informed by its observed centre of light [9 parameters].
 - The source galaxy's light is a point `PointSourceFlux` [3 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""
model_path = path.join("scripts", "clusters", "chaining", "models")
model_file = path.join(model_path, "lens_x3__source_x1.json")

model = af.Collection.from_json(file=model_file)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.
"""
search_1 = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[1]_point_source",
    unique_tag=dataset_name,
    nlive=50,
    number_of_cores=1,
)

analysis = al.AnalysisPointSource(
    point_source_dict=point_source_dict, solver=positions_solver
)

result_1 = search_1.fit(model=model, analysis=analysis)

"""
__Masking (Search 2)__

The model-fit to imaging data requires a `Mask2D`. 

Note how this has a radius of 9.0", much larger than most example lenses!
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=9.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Model (Search 2)__

We use the results of search 1 to create the lens model fitted in search 2, where:

 - There are again three lens galaxy's with `SphSersic` light profiles [15 parameters: centres initialized from model].
 - The three lens galaxy's have `SphIsothermal` mass distributions [9 parameters: priors initialized from search 1].
 - The source-galaxy's light uses a `EllSersic` light profile [7 parameters: centre initialized from search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=31.

The term `model` below passes the source model as model-components that are to be fitted for by the 
non-linear search. We pass the `lens` as a `model`, so that we can use the mass model inferred by search 1. The source
does not use any priors from the result of search 1.
"""
lens_0 = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp.SphSersic, mass=al.mp.SphIsothermal
)
lens_0.bulge.centre = model.galaxies.lens_0.mass.centre
lens_0.mass = result_1.model.galaxies.lens_0.mass

lens_1 = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp.SphSersic, mass=al.mp.SphIsothermal
)
lens_1.bulge.centre = model.galaxies.lens_1.mass.centre
lens_1.mass = result_1.model.galaxies.lens_1.mass

lens_2 = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp.SphSersic, mass=al.mp.SphIsothermal
)
lens_2.bulge.centre = model.galaxies.lens_2.mass.centre
lens_2.mass = result_1.model.galaxies.lens_2.mass

source_0 = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.EllSersic)
source_0.bulge.centre = result_1.model.galaxies.source_0.point_0.centre

model = af.Collection(
    galaxies=af.Collection(
        lens_0=lens_0, lens_1=lens_1, lens_2=lens_2, source_0=source_0
    )
)

"""
__Analysis + Positions__

In this example we update the positions between searches, where the positions correspond to the (y,x) locations of the 
lensed source's multiple images. When a model-fit uses positions, it requires them to trace within a threshold value of 
one another for every mass model sampled by the non-linear search. If they do not, the model is discarded and resampled. 

Below, we use the point source dictionary positions and a threshold double the resolution of the data, which should be
sufficient for this analysis.
"""
settings_lens = al.SettingsLens(positions_threshold=2.0 * imaging.pixel_scales[0])

analysis = al.AnalysisImaging(
    dataset=imaging,
    positions=point_source_dict["point_0"].positions,
    settings_lens=settings_lens,
)

"""
__Search + Model-Fit__

We now create the non-linear search and perform the model-fit using this model.
"""
search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[2]__imaging",
    unique_tag=dataset_name,
    nlive=100,
)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

In this example, we passed used prior passing to initialize a lens model using a point source dataset and passed this 
a second fit which fitted the full `Imaging` dataset. 

This circumvented the challenge with initializing a high dimensionality complex lens model to `Imaging` data where
the computational run time is slower.
"""
