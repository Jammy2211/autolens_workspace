import os
import numpy as np

### INTERFEROMETER ###

# This runner performs an analysis using an interferometer pipeline, which fits data from an interferometer (e.g. ALMA
# or a radio telescope) in the uv-plane. The lens modeling in interferometer pipelines are nearly identical to the
# 'no_lens_light' pipelines used for imaging data, but with setting specific to interferometer data.

### WELCOME ###

# Welcome to the pipeline runner, which loads a strong lens dataset and analyses it using a lens modeling pipeline.

# This script uses an 'advanced' pipeline. I'll be assuming that you are familiar with 'beginner' and 'intermediate'
# pipelines, so if anything isn't clear check back to the their runners and pipelines!

# First, lets consider some aspects of beginner and intermediate pipelines that were sub-optimal:

# - They often repeated the same phases to initialize the lens model. For example, source inversion pipelines using
#   hyper-features always repeated the same 4 phases to initialize the inversion (use a magnification based pixelization
#   and then surface-brightness based one). This lead to lots of repetition and wasted processing time!

# - Tweaking a pipeline to slightly change the model it fitted often required a rerun of the entire pipeline or
#   for one to create a new pipeline that slightly changed its behaviour.

# Advanced address these problems using pipeline composition, whereby multiple different pipelines each focused on
# fitting a specific aspect of the lens model are added together. The results of the initial pipelines are then reused
# when changing the model fitted in later pipelines. For example, we may initialize a source inversion using such a
# pipeline and use this to fit a variety of different mass models with different pipelines.

### SLAM (Source, Light and Mass) ###

# Advanced pipelines are written following the SLAM method, whereby we first initialize the source fit to a lens,
# followed by the lens's light and then the lens's mass.

# If you think back to the beginner and intermediate pipelines this is exactly what they di. We'd get the inversion
# up and running first, then refine the lens's light model and finally its mass. Advanced pipelines simply use
# separate pipelines to do this, each of which has features that enable more customization of the model that is fitted.

### THIS RUNNER ###

# Using two source pipelines and a mass pipeline we will fit a power-law mass model and source using a pixelized
# inversion.

# We'll use the example pipelines:
# 'autolens_workspace/pipelines/advanced/no_lens_light/source/parametric/lens_sie__source_sersic.py'.
# 'autolens_workspace/pipelines/advanced/no_lens_light/source/inversion/from_parametric/lens_sie__source_inversion.py'.
# 'autolens_workspace/pipelines/advanced/no_lens_light/mass/power_law/lens_power_law__source.py'.

# Check them out now for a detailed description of the analysis!

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
dataset_label = "interferometer"
dataset_name = "lens_sie__source_sersic"

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# This loads the interferometer dataset,.
interferometer = al.interferometer.from_fits(
    visibilities_path=dataset_path + "visibilities.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    uv_wavelengths_path=dataset_path + "uv_wavelengths.fits",
)

# The visibilities mask is used to mask out any visilbiities in our data we don't want to fit. Lets assume we'll
# fit the entire data-set.
visibilities_mask = np.full(fill_value=False, shape=interferometer.visibilities.shape)

# Next, we create the real-space mask which defines the real-space grid the model images of the strong lens system
# are created on and used to transform to the uv-plane model data.
real_space_mask = al.mask.circular(shape_2d=(151, 151), pixel_scales=0.1, radius=3.0)

# Make a quick subplot to make sure the data looks as we expect.
aplt.interferometer.subplot_interferometer(interferometer=interferometer)


### PIPELINE SETUP ###

# Advanced pipelines still use general setup, which customize the hyper-mode features and inclusion of a shear.

general_setup = al.setup.General(
    hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
)

source_setup = al.setup.Source(
    pixelization=al.pix.VoronoiBrightnessImage, regularization=al.reg.AdaptiveBrightness
)

mass_setup = al.setup.Mass(fix_lens_light=False)

setup = al.setup.Setup(general=general_setup, source=source_setup, mass=mass_setup)

# We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

### SOURCE ###

from pipelines.advanced.interferometer.source.parametric import lens_sie__source_sersic
from pipelines.advanced.interferometer.source.inversion.from_parametric import (
    lens_sie__source_inversion,
)

pipeline_source__parametric = lens_sie__source_sersic.make_pipeline(
    setup=setup,
    real_space_mask=real_space_mask,
    phase_folders=["advanced", dataset_label, dataset_name],
)

pipeline_source__inversion = lens_sie__source_inversion.make_pipeline(
    setup=setup,
    real_space_mask=real_space_mask,
    phase_folders=["advanced", dataset_label, dataset_name],
)

### MASS ###

from pipelines.advanced.interferometer.mass.power_law import lens_power_law__source

# The mass setup for this pipeline is shown below, which define:

mass_setup = al.setup.Mass()

pipeline_mass__power_law = lens_power_law__source.make_pipeline(
    setup=setup,
    real_space_mask=real_space_mask,
    phase_folders=["advanced", dataset_label, dataset_name],
)

### PIPELINE COMPOSITION AND RUN ###

# We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
# information throughout the analysis to later phases.

pipeline = (
    pipeline_source__parametric + pipeline_source__inversion + pipeline_mass__power_law
)

pipeline.run(dataset=interferometer, mask=visibilities_mask)
