# %%
"""
__WELCOME__ 

Welcome to the pipeline runner, which loads a strong lens dataset and analyses it using a lens modeling pipeline.

This script uses an 'intermediate' pipeline. I'll be assuming that you are familiar with 'beginner' pipelines, so if
anything isn't clear check back to the beginner runners and pipelines!

Intermediate runners and pipelines introduce PyAutoLens's 'hyper-mode'. Hyper-mode passes the best-fit model-image
of previous phases in a pipeline to later phases, and then uses this model image (called the 'hyper image') to:

- Adapt a pixelization's grid to the surface-brightness of the source galaxy.
- Adapt the _Regularization_ scheme to the surface-brightness of the source galaxy.
- Scale the noise in regions of the image where the model give a poor fit (in both the lens and source galaxies).
- Include uncertanties in the data-reduction into the model, such as the background sky level.

__THIS RUNNER __

describes how to set up and run a pipeline which uses hyper-mode. A full description of hyper-model is
given in chapter 5 of the HowToLens lecture series.

Using a pipeline composed of three phases we fit an  _EllipticalIsothermal_ mass model and source using a pixelized _Inversion_.

We'll use the example pipeline:
'autolens_workspace/pipelines/intermediate/no_lens_light/lens_sie__source_inversion.py'.

Check it out now for a detailed description of the analysis, including hyper-mode!
"""

# %%
""" AUTOFIT + CONFIG SETUP """

# %%
from autoconf import conf
import autofit as af

# %%
"""Setup the path to the autolens_workspace, using a relative directory name."""

# %%
"""Use the WORKSPACE environment variable to determine the path to the autolens workspace."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""Use this path to explicitly set the config path and output path."""

# %%
from autoconf import conf

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "mass_sie__source_sersic"
pixel_scales = 0.1

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/mass_sie__source_sersic'
"""

# %%
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

# %%
"""Using the dataset path, load the data (image, noise-map, PSF) as an _Imaging_ object from .fits files."""
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

# %%
"""Next, we create the mask we'll fit this data-set with."""
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# %%
"""Make a quick subplot to make sure the data looks as we expect."""
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Settings__

*SettingsPhase* behave as they did in normal pipelines.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__PIPELINE SETUP__

We again use the SetupPipeline to customize the pipeline's behaviour, which include input that control the behaviour of 
hyper-mode, specifically:

- If hyper-galaxies are used to scale the noise in each component of the image (default True)
- If the level of background noise is modeled throughout the pipeline (default True)
- If the background sky is modeled throughout the pipeline (default False)

We are now able to use the *VoronoiBrightnessImage* _Pixelization_ and *AdaptiveBrightness* _Regularization_ scheme, which 
in hyper-mode to adapt the _Pixelization_ and regularizatioon to the morphology of the lensed source galaxy. 
"""

# %%
setup = al.SetupPipeline(
    hyper_galaxies=True,
    hyper_background_noise=False,
    hyper_image_sky=False,  # <- By default this feature is off, as it rarely changes the lens model.
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
    folders=["hyper", dataset_type, dataset_label, dataset_name],
)

# %%
"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!
"""

# %%
from autolens_workspace.advanced.hyper.pipelines.no_lens_light import (
    lens_sie__source_inversion,
)

pipeline = lens_sie__source_inversion.make_pipeline(setup=setup, settings=settings)

# %%
"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

pipeline.run(dataset=imaging, mask=mask)
