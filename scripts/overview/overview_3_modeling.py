"""
Overview: Modeling
------------------

Lens modeling is the process of taking data of a strong lens (e.g. imaging data from the Hubble Space Telescope or
interferometer data from ALMA) and fitting it with a lens model, to determine the `LightProfile`'s and `MassProfile`'s
that best represent the observed strong lens.

Lens modeling with **PyAutoLens** uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import **PyAutoFit** separately to **PyAutoLens**
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autolens as al
import autolens.plot as aplt

import autofit as af

"""
__Dataset__

In this example, we fit simulated imaging of the strong lens SLACS1430+4105. First, lets load this
imaging dataset and plot it.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Masking__

We next mask the dataset, to remove the exterior regions of the image that do not contain emission from the lens or
source galaxy.

Note how when we plot the `Imaging` below, the figure now zooms into the masked region.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot()

"""
__Model__

We compose the lens model that we fit to the data using `Model` objects. These behave analogously to `Galaxy`
objects but their  `LightProfile` and `MassProfile` parameters are not specified and are instead determined by a
fitting procedure.

We will fit our strong lens data with two galaxies:

- A lens galaxy with a `EllSersic` `LightProfile` representing a bulge and
  `EllIsothermal` `MassProfile` representing its mass.
- A source galaxy with an `EllExponential` `LightProfile` representing a disk.

The redshifts of the lens (z=0.5) and source(z=1.0) are fixed.
"""
lens_galaxy_model = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp.EllSersic, mass=al.mp.EllIsothermal
)

source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, disk=al.lp.EllExponential)

"""
We combine the lens and source model galaxies above into a `Collection`, which is the model we will fit. Note how
we could easily extend this object to compose highly complex models containing many galaxies.
"""
model = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)

"""
__Non-linear Search__

We now choose the non-linear search, which is the fitting method used to determine the set of `LightProfile`
and `MassProfile` parameters that best-fit our data.

In this example we use `dynesty` (https://github.com/joshspeagle/dynesty), a nested sampling algorithm that is
very effective at lens modeling.
"""
search = af.DynestyStatic(name="overview_modeling")

"""
__Analysis__

We next create an `AnalysisImaging` object, which contains the `log likelihood function` that the non-linear search 
calls to fit the lens model to the data.
"""
analysis = al.AnalysisImaging(dataset=imaging)

"""
__Model-Fit__

To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
dynesty samples, model parameters, visualization) to hard-disk.

Once running you should checkout the `autolens_workspace/output` folder, which is where the results of the search are 
written to hard-disk (in the `overview_modeling` folder) on-the-fly. This includes lens model parameter estimates with 
errors non-linear samples and the visualization of the best-fit lens model inferred by the search so far. 
"""
result = search.fit(model=model, analysis=analysis)

"""
__Results__

Whilst navigating the output folder, you may of noted the results were contained in a folder that appears as a random
collection of characters. 

This is the model-fit's unique identifier, which is generated based on the model, search and dataset used by the fit. 
Fitting an identical model, search and dataset will generate the same identifier, meaning that rerunning the script 
will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset, a new 
unique identifier will be generated, ensuring that the model-fit results are output into a separate folder.

The fit above returns a ``Result`` object, which contains the maximum log likelihood ``Tracer`` and ``FitImaging``
objects and information on the posterior estimated by Dynesty, all of which can easily be plotted.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=imaging.grid
)
tracer_plotter.subplot_tracer()

fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

"""
In fact, this ``Result`` object contains the full posterior information of our non-linear search, including all
parameter samples, log likelihood values and tools to compute the errors on the lens model.

The script `autolens_workspace/notebooks/imaging/modeling/result.py` contains a full description of all information 
contained in a ``Result``.

__Wrap Up__

A more detailed description of lens modeling with **PyAutoLens**'s is given in chapter 2 of the **HowToLens** 
tutorials, which I strongly advise new users check out!
"""
