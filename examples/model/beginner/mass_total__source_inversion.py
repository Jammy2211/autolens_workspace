"""
__Example: Modeling__

To fit a lens model to a dataset, we must perform lens modeling, which uses a `NonLinearSearch` to fit many
different tracers to the dataset.

Model-fitting is handled by our project **PyAutoFit**, a probabilistic programming language for non-linear model
fitting. The setting up of configuration files is performed by our project **PyAutoConf**. we'll need to import
both to perform the model-fit.
"""

"""
In this example script, we fit `Imaging` of a strong lens system where:

 - The lens `Galaxy`'s light is omitted (and is not present in the simulated data).
 - The lens `Galaxy`'s total mass distribution is modeled as an `EllipticalIsothermal`.
 - The source `Galaxy`'s surface-brightness is modeled using an `Inversion`.

An `Inversion` reconstructs the source`s light using a pixel-grid, which is regularized using a prior that forces
this reconstruction to be smooth. This uses `Pixelization` and `Regularization` objects and in this example we will
use their simplest forms, a `Rectangular` `Pixelization` and `Constant` `Regularization`.scheme.

Inversions are covered in detail in chapter 4 of the **HowToLens** lectures.
"""

"""
Load the strong lens dataset `mass_sie__source_sersic` `from .fits files, which is the dataset we will
use to perform lens modeling.

This is the same dataset we fitted in the `autolens/intro/fitting.py` example.
"""

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
    positions_path=path.join(dataset_path, "positions.dat"),
)

"""The model-fit also requires a mask defining the regions of the image we fit the lens model to the data."""

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

"""
__Phase__

To perform lens modeling, we create a `PhaseImaging` object, which comprises:

   - The `GalaxyModel`'s used to fit the data.
   - The `SettingsPhase` which customize how the model is fitted to the data.
   - The `NonLinearSearch` used to sample parameter space.
   
Once we have create the phase, we `run` it by passing it the data and mask.
"""

"""
__Model__

We compose our lens model using `GalaxyModel` objects, which represent the galaxies we fit to our data. In this 
example our lens mooel is:

 - An `EllipticalIsothermal` `MassProfile`.for the lens `Galaxy`'s mass (5 parameters).
 - A `Rectangular` `Pixelization`.which reconstructs the source `Galaxy`'s light. We will fix its resolution to 
   30 x 30 pixels, which balances fast-run time with sufficient resolution to reconstruct its light. (0 parameters).
 - A `Constant` `Regularization`.scheme which imposes a smooothness prior on the source reconstruction (1 parameter). 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=1. 
 
It is worth noting the `Pixelization` and `Regularization` use significantly fewer parameter (1 parameters) than 
fitting the source using `LightProfile`'s (7+ parameters). 

NOTE: By default, **PyAutoLens** assumes the image has been reduced such that the lens galaxy centre is at (0.0", 0.0"),
with the priors on the lens `MassProfile` coordinates set accordingly. if for your dataset the lens is not centred at 
(0.0", 0.0"), we recommend you reduce your data so it is (see `autolens_workspace/preprocess`). Alternatively, you can 
manually override the priors (see `autolens_workspace/examples/customize/priors.py`).
"""

lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(30, 30)),
    regularization=al.reg.Constant,
)

"""
__Settings__

Next, we specify the `SettingsPhaseImaging`, which describe how the model is fitted to the data in the log likelihood
function. Below, we specify:
 
 - That a regular `Grid` is used to fit create the model-image when fitting the data 
      (see `autolens_workspace/examples/grids.py` for a description of grids).
 - The sub-grid size of this grid.

We specifically specify the grid that is used to perform the `Inversion`. In **PyAutoLens** it is possible to fit
data simultaneously with `LightProfile`'s and an `Inversion`. Each fit uses a different grid, which are specified 
independently.

`Inversion`'s suffer a problem where they reconstruct unphysical lens models, where the reconstructed soruce appears
as a demagnified reconstruction of the lensed source. These are covered in more detail in chapter 4 of **HowToLens**. 
To prevent these solutions impacting this fit we use position thresholding, which is describe fully in the 
script `autolens_workspace/examples/model/customize/positions.py`, Therefore, we also specify:

 - A positions_threshold of 0.5, meaning that the four (y,x) coordinates specified by our positions must trace
   within 0.5" of one another in the source-plane for a mass model to be accepted. If not, it is discarded and
   a new model is sampled.

The threshold of 0.5" is large. For an accurate lens model we would anticipate the positions trace within < 0.01" of
one another. However, we only want the threshold to aid the `NonLinearSearch` with the generation of the initial 
mass models. 

Different `SettingsPhase` are used in different example model scripts and a full description of all `SettingsPhase` 
can be found in the example script `autolens/workspace/examples/model/customize/settings.py` and the following 
link -> <link>
"""

settings_masked_imaging = al.SettingsMaskedImaging(
    grid_inversion_class=al.Grid, sub_size=2
)
settings_lens = al.SettingsLens(positions_threshold=0.5)

settings = al.SettingsPhaseImaging(
    settings_masked_imaging=settings_masked_imaging, settings_lens=settings_lens
)

"""
__Search__

The lens model is fitted to the data using a `NonLinearSearch`, which we specify below. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/), with:

 - 50 live points.

The script `autolens_workspace/examples/model/customize/non_linear_searches.py` gives a description of the types of
non-linear searches that can be used with **PyAutoLens**. If you do not know what a `NonLinearSearch` is or how it 
operates, I recommend you complete chapters 1 and 2 of the HowToLens lecture series.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/beginner/mass_sie__source_sersic/phase_mass[sie]_source[inversion]`.
"""

search = af.DynestyStatic(
    path_prefix=path.join("examples", "beginner", dataset_name),
    name="phase_mass[sie]_source[inversion]",
    n_live_points=50,
)

"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.
"""

phase = al.PhaseImaging(
    search=search,
    galaxies=af.CollectionPriorModel(lens=lens, source=source),
    settings=settings,
)

"""
We can now begin the fit by passing the dataset and mask to the phase, which will use the `NonLinearSearch` to fit
the model to the data. 

The fit outputs visualization on-the-fly, so checkout the path 
`/path/to/autolens_workspace/output/examples/phase_mass[sie]_source[inversion]` to see how your fit is doing!
"""

result = phase.run(dataset=imaging, mask=mask)

"""
The phase above returned a result, which, for example, includes the lens model corresponding to the maximum
log likelihood solution in parameter space.
"""

print(result.max_log_likelihood_instance)

"""
It also contains instances of the maximum log likelihood Tracer and FitImaging, which can be used to visualize
the fit.
"""

aplt.Tracer.subplot_tracer(
    tracer=result.max_log_likelihood_tracer, grid=mask.geometry.masked_grid_sub_1
)
aplt.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

"""
Checkout `/path/to/autolens_workspace/examples/model/results.py` for a full description of the result object.
"""
