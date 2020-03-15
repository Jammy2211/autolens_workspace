import os

### WELCOME ###

# Welcome to the pipeline runner, which loads a strong lens dataset and analyses it using a lens modeling pipeline.

# This script uses an 'intermediate' pipeline. I'll be assuming that you are familiar with 'beginner' pipelines, so if
# anything isn't clear check back to the beginner runners and pipelines!

# Intermediate runners and pipelines introduce PyAutoLens's 'hyper-mode'. Hyper-mode passes the best-fit model-image
# of previous phases in a pipeline to later phases, and then uses this model image (called the 'hyper image') to:

# - Adapt a pixelization's grid to the surface-brightness of the source galaxy.
# - Adapt the regularization scheme to the surface-brightness of the source galaxy.
# - Scale the noise in regions of the image where the model give a poor fit (in both the lens and source galaxies).
# - Include uncertanties in the data-reduction into the model, such as the background sky level.

# This runner describes how to set up and run a pipeline which uses hyper-mode. A full description of hyper-model is
# given in chapter 5 of the HowToLens lecture series.

### THIS RUNNER ###

# Using a pipeline composed of three phases we will fit an SIE mass model and source using a pixelized inversion.

# We'll use the example pipeline:
# 'autolens_workspace/pipelines/intermediate/no_lens_light/lens_sie__source_inversion.py'.

# Check it out now for a detailed description of the analysis, including hyper-mode!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al
import autolens.plot as aplt

# Specify the dataset label and name, which we use to determine the path we load the data from.
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"
pixel_scales = 0.1

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# Using the dataset path, load the data (image, noise-map, PSF) as an imaging object from .fits files.
imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

# Next, we create the mask we'll fit this data-set with.
mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# Make a quick subplot to make sure the data looks as we expect.
aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)

### PIPELINE SETUP ###

# In the beginner runners, we used the'pipeline_setup' to customize the pipeline's behaviour. Intermediate pipeline
# use these pipeline setup to control the behaviour of hyper-mode, specifically:

# - If hyper-galaxies are used to scale the noise in each component of the image (default True)
# - If the level of background noise is modeled throughout the pipeline (default True)
# - If the background sky is modeled throughout the pipeline (default False)

general_setup = al.setup.General(
    hyper_galaxies=True,
    hyper_background_noise=True,
    hyper_image_sky=False,  # <- By default this feature is off, as it rarely changes the lens model.
)

# Source and mass setup are required as in the beginne pipelines.

source_setup = al.setup.Source(
    pixelization=al.pix.VoronoiBrightnessImage,
    # <- These behave as they did for beginner pipelines.
    regularization=al.reg.AdaptiveBrightness,
)

mass_setup = al.setup.Mass(no_shear=False)

### PIPELINE SETUP + RUN ###

# First, we group all of the setup above in a pipeline setup object.

setup = al.setup.Setup(general=general_setup, source=source_setup, mass=mass_setup)

# To run a pipeline we import it from the pipelines folder, make it and pass the lens data to its run function. Now we
# are using hyper-features we can use the VoronoiBrightnessImage pixelization and AdaptiveBrightness regularization.

from pipelines.intermediate.no_lens_light import lens_sie__source_inversion

pipeline = lens_sie__source_inversion.make_pipeline(
    setup=setup, phase_folders=["intermediate", dataset_label, dataset_name]
)

pipeline.run(dataset=imaging, mask=mask)
