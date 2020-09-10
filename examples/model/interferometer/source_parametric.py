# %%

"""
__Example: Interferometer Source Parametric__

To fit a lens model to an interferometer dataset, we again perform lens modeling using a non-linear search algorithm.
However, unlike CCD _Imaging_ data, we fit the lens model in Fourier space, or the 'uv-plane', which circumvents issues
that arise when trying to fit CLEANED images of interferometer data.

A big challenge when fitting interferometer datasets is the huge quantity of data. Very long baseline ALMA or JVLA
observations observe in excess of *millions* of visibilities, which can make certain approaches to modeling
interferometer data extremely slow and expensive.

In this example, we fit an interferometer dataset consisting of 1 million visibilities, assuming a parametric
_EllipticalSersic_ model for the source. This analysis would be very expensive if a direct Fourier transform is used to
transform the lens model-image from  real-space to Fourier space and compare with the visibilities in the uv-plane.

Instead, **PyAutoLens** uses the non-uniform fast Fourier transform (NUFFT) of the package PyNUFFT
(https://github.com/jyhmiinlin/pynufft)  which ensures we can fit > 1 million visibilities in an efficient manner!
"""

# %%
"""
In this example script, we fit interferometer data of a strong lens system where:

 - The lens galaxy's _LightProfile_ is omitted (and is not present in the simulated data).
 - The lens galaxy's _MassProfile_ is modeled as an _EllipticalIsothermal_.
 - The source galaxy's _LightProfile_ is modeled as an _EllipticalSersic_.

"""

# %%
"""Setup the path to the autolens workspace, using pyprojroot to determine it automatically."""

# %%
"""Use the WORKSPACE enviroment variable to determine the workspace path."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""
Load the strong lens dataset 'mass_sie__source_sersic' 'from .fits files.

Unlike the other example scripts, we use the _Interferometer_ class to load this dataset, passing it paths to the .fits
files containing its visibilities, noise-map and uv_wavelengths.
"""

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

dataset_type = "interferometer"
dataset_name = "mass_sie__source_sersic__2"
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_name}"

interferometer = al.Interferometer.from_fits(
    visibilities_path=f"{dataset_path}/visibilities.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    uv_wavelengths_path=f"{dataset_path}/uv_wavelengths.fits",
)

aplt.Interferometer.subplot_interferometer(interferometer=interferometer)

# %%
"""
The perform a fit, we need two masks, firstly a ‘real-space mask’ which defines the grid the image of the lensed 
source galaxy is evaluated using.
"""

# %%
real_space_mask = al.Mask.circular(shape_2d=(200, 200), pixel_scales=0.05, radius=3.0)

# %%
"""
We also need a ‘visibilities mask’ which defines which visibilities are omitted from the chi-squared evaluation.
"""

# %%
visibilities_mask = np.full(fill_value=False, shape=interferometer.visibilities.shape)

# %%
"""
__Phase__

To perform lens modeling, we create a *PhaseInterferometer* object, which comprises:

   - The _GalaxyModel_'s used to fit the data.
   - The *SettingsPhase* which customize how the model is fitted to the data.
   - The *NonLinearSearch* used to sample parameter space.
   
Once we have create the phase, we 'run' it by passing it the data and mask.
"""

# %%
"""
__Model__

We compose our lens model using _GalaxyModel_ objects, which represent the galaxies we fit to our data. In this 
example our lens mooel is:

 - An _EllipticalIsothermal_ _MassProfile_ for the lens galaxy's mass (5 parameters).
 - An _EllipticalSersic_ _LightProfile_ for the source galaxy's light (7 parameters).

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12.
"""

# %%
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, sersic=al.lp.EllipticalSersic)

# %%
"""
__Settings__

Next, we specify the *SettingsPhaseInterferometer*, which describes how the model is fitted to the data in the log 
likelihood function. Below, we specify:
 
 - That a regular _Grid_ is used to fit create the model-image (in real space) when fitting the data 
   (see 'autolens_workspace/examples/grids.py' for a description of grids).
 - The sub-grid size of this real-space grid.
 - The method used to Fourier transform this real-space image of the strong lens to the uv-plane, to compare directly
   to the visiblities. In this example, we use a non-uniform fast Fourier transform, which operates efficiently
   on interferometer datasets containing ~1-10 million visibilities.
"""

# %%
settings_masked_interferometer = al.SettingsMaskedInterferometer(
    grid_class=al.Grid, sub_size=2, transformer_class=al.TransformerNUFFT
)

settings = al.SettingsPhaseInterferometer(
    masked_interferometer=settings_masked_interferometer
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

 '/autolens_workspace/output/examples/beginner/mass_sie__source_sersic/phase__mass_sie__source_sersic'.
"""

# %%
phase = al.PhaseInterferometer(
    phase_name="phase__mass_sie__source_sersic",
    real_space_mask=real_space_mask,
    folders=["examples", "interferometer", dataset_name],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

# %%
"""
We can now begin the fit by passing the dataset and visibilties mask to the phase, which will use the non-linear search 
to fit the model to the data. 

The fit outputs visualization on-the-fly, so checkout the path 
'/path/to/autolens_workspace/output/examples/phase__mass_sie__source_sersic' to see how your fit is doing!
"""

# %%
result = phase.run(dataset=interferometer, mask=visibilities_mask)

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
aplt.Tracer.subplot_tracer(
    tracer=result.max_log_likelihood_tracer,
    grid=real_space_mask.geometry.masked_grid_sub_1,
)
aplt.FitImaging.subplot_fit_imaging(fit=result.max_log_likelihood_fit)

# %%
"""
Checkout '/path/to/autolens_workspace/examples/model/results.py' for a full description of the result object.
"""
