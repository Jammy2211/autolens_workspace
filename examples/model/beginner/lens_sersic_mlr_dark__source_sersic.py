# %%
"""
__Example: Modeling__

To fit a lens model to a dataset, we must perform lens modeling, which uses a non-linear search algorithm to fit many
different tracers to the dataset.

Model-fitting is handled by our project **PyAutoFit**, a probablistic programming language for non-linear model
fitting. The setting up on configuration files is performed by our project **PyAutoConf**. We'll need to import
both to perform the model-fit.
"""

# %%
"""
In this example script, we will fit imaging of a strong lens system where:

 - The lens galaxy's _LightProfile_ is fitted with an _EllipticalSersic_.
 - The lens galaxy's stellar _MassProfile_ is fitted with the EllipticalSersic of the 
      _LightProfile_, where it is converted to a stellar mass distribution via a constant mass-to-light ratio.
 - The lens galaxy's nfw _MassProfile_ is fitted with a _SphericalNFW_.
 - The source galaxy's _LightProfile_ is fitted with an _EllipticalSersic_.   
"""

# %%
"""Setup the path to the autolens workspace, using the project pyprojroot which determines it automatically."""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# %%
"""
We use this path to set:
    config_path:
        Where PyAutoLens configuration files are located. The default location is '/path/to/autolens_workspace/config'. 
        They control many aspects of PyAutoLens (visualization, model priors, etc.). Feel free to check them out!.
    
    output-path: 
        Where the output of the non-linear search and model-fit are stored on your hard-disk. The default location 
        is '/path/to/autolens_workspace/output.
"""

# %%
from autoconf import conf

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
"""
Load the strong lens dataset 'lens_sersic_mlr_nfw__source_sersic' 'from .fits files, which is the dataset 
we will use to perform lens modeling.

This is the same dataset we fitted in the 'fitting.py' example.
"""

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "stellar_and_dark"
dataset_name = "lens_sersic_mlr_nfw__source_sersic"
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=0.1,
)

# %%
"""
The model-fit also requires a mask, which defines the regions of the image we use to fit the lens model to the data.
"""

# %%
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Phase__

To perform lens modeling, we create a *PhaseImaging* object, which comprises:

   - The _GalaxyModel_'s used to fit the data.
   - The *PhaseSettings* which customize how the model is fitted to the data.
   - The *NonLinearSearch* used to sample parameter space.

Once we have create the phase, we 'run' it by passing it the data and mask.
"""

# %%
"""
__Model__

We compose our lens model using _GalaxyModel_ objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - An EllipticalSersic *LightAndMassProfile* for the lens galaxy's light and mass (8 parameters) [note that we use 
      the 'al.lmp' module to create this, signifying it represents both the light and mass].
 - A SphericalNFW _MassProfile_ for the lens galaxy's nfw matter (4 parameters).
 - An Elliptical_EllipticalSersic_ _LightProfile_ for the source galaxy's mass (7 parameters).

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.
"""

# %%
lens = al.GalaxyModel(
    redshift=0.5, stellar=al.lmp.EllipticalSersic, dark=al.mp.SphericalNFW
)
source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

# %%
"""
__Settings__

Next, we specify the *PhaseSettingsImaging*, which describe how the model is fitted to the data in the log likelihood
function. Below, we specify:

 - That a *GridInterpolate* is used to fit create the model-image when fitting the data 
      (see 'autolens_workspace/examples/grids.py' for a description of grids).
 - The pixel-scale of this interpolation grid.

The deflection angle calculation of the _EllipticalSersic_ _MassProfile_ requires numerical integration and is
computationally more expensive than most mass profiles. For this reason, we use a *GridInterpolate* grid instead of
the *Grid* we use in most other examples, which limits the deflection angle calculation to a grid of reduced resolution
and interpolates the results to the native-resolution grid. 

A description of the *GridIterpolate* object can be found in the script 'autolens_workspace/examples/grids.py'.

Different *PhaseSettings* are used in different example model scripts and a full description of all *PhaseSettings* 
can be found in the example script 'autolens/workspace/examples/model/customize/settings.py' and the following 
link -> <link>
"""

# %%
settings = al.PhaseSettingsImaging(
    grid_class=al.GridInterpolate, pixel_scales_interp=0.1
)

# %%
"""
__Search__

The lens model is fitted to the data using a *NonLinearSearch*, which we specify below. In this example, we use the
nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/), with:

 - 50 live points.

The script 'autolens_workspace/examples/model/customize/non_linear_searches.py' gives a description of the types of
non-linear searches that can be used with **PyAutoLens**. If you do not know what a non-linear search is or how it 
operates, I recommend you complete chapters 1 and 2 of the HowToLens lecture series.
"""

# %%
search = af.DynestyStatic(n_live_points=50)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/beginner/lens_sersic_mlr_nfw__source_sersic/
     phase__lens_sersic_mlr_nfw__source_sersic'.
"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__lens_sersic_mlr_nfw__source_sersic",
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
    folders=["examples", "beginner", dataset_name],
)

# %%
"""
We can now begin the fit by passing the dataset and mask to the phase, which will use the non-linear search to fit
the model to the data. 

The fit outputs visualization on-the-fly, so checkout the path 
'/path/to/autolens_workspace/output/examples/phase__lens_sersic_mlr_nfw__source_sersic' to see how your fit is doing!
"""

# %%
result = phase.run(dataset=imaging, mask=mask)

# %%
"""
The phase above returned a result, which, for example, includes the lens model corresponding to the maximum
log likelihood solution in parameter space.
"""

# %%
print(result.max_log_likelihood_instance)

# %%
"""
It also contains instances of the maximum log likelihood Tracer and FitImaging, which can be used to visualize
the fit.
"""

# %%
aplt.Tracer.subplot_tracer(tracer=result.max_log_likelihood_tracer, grid=imaging.grid)
aplt.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

# %%
"""
Checkout '/path/to/autolens_workspace/examples/model/results.py' for a full description of the result object.
"""
