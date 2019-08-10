import autofit as af
from autolens.data.array import mask as msk
from autolens.data.instrument import ccd
from autolens.data.plotters import ccd_plotters
from autolens.pipeline import pipeline as pl

import os

# Welcome to the advanced pipeline runner! This script is identical to the
# 'runners/simple/runner__lens_sersic_sie__source_sersic.py' script, except at the end when we add pipelines together. So,
# if you already know how the simple runners work, jump ahead to our pipeline imports. If you don't, I recommmend you
# checout the 'simple' pipelines, before using this script.

### The code between the dashed ---- lines is identical to 'runners/simple/runner__lens_sersic_sie__source_sersic.py' ###

# ----------------------------------------------------------------------------------------------------------

# Welcome to the pipeline runner. This tool allows you to load strong lens instrument, and pass it to pipelines for a
# PyAutoLens analysis. To show you around, we'll load up some example instrument and run it through some of the example
# pipelines that come distributed with PyAutoLens.

# The runner is supplied as both this Python script and a Juypter notebook. Its up to you which you use - I personally
# prefer the python script as provided you keep it relatively small, its quick and easy to comment out different lens
# names and pipelines to perform different analyses. However, notebooks are a tidier way to manage visualization - so
# feel free to use notebooks. Or, use both for a bit, and decide your favourite!

# The pipeline runner is fairly self explanatory. Make sure to checkout the pipelines in the
#  workspace/pipelines/examples/ folder - they come with detailed descriptions of what they do. I hope that you'll
# expand on them for your own personal scientific needs

# Setup the path to the workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

# Create the path to the data folder in your workspace.
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data"]
)

# It is convenient to specify the instrument type and data name as a string, so that if the pipeline is applied to multiple
# images we don't have to change all of the path entries in the load_ccd_data_from_fits function below.
data_type = "example"
data_name = (
    "lens_sersic_sie__source_sersic"
)  # Example simulated image with lens light emission and a source galaxy.
pixel_scale = 0.1

# data_name = 'slacs1430+4105' # Example HST imaging of the SLACS strong lens slacs1430+4150.
# pixel_scale = 0.03

# Create the path where the instrument will be loaded from, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=data_path, folder_names=[data_type, data_name]
)

# This loads the CCD imaging data, as per usual.
ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    pixel_scale=pixel_scale,
)

# We need to define and pass our mask to the hyper pipeline from the beginning.
mask = msk.Mask.circular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)

# Plot CCD before running.
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask)


# Running a pipeline is easy, we simply import it from the pipelines folder and pass the lens instrument to its run function.
# Below, we'll' use a 3 phase example pipeline to fit the instrument with a parametric lens light, mass and source light
# profile. Checkout 'workspace/pipelines/examples/lens_light_and_x1_source_parametric.py' for a full description of
# the pipeline.

# The phase folders input determines the output directory structure of the pipeline, for example the input below makes
# the directory structure:
# 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/' or
# 'autolens_workspace/output/example/lens_light_and_x1_source/lens_light_and_x1_source_parametric/'

# For large samples of images, we can therefore easily group lenses that are from the same sample or modeled using the
# same pipeline.

# --------------------------------------------------------------------------------------------------------

### HYPER PIPELINE SETTINGS ###

# In the advanced pipelines, we defined pipeline settings which controlled various aspects of the pipelines, such as
# the model complexity and assumtpions we made about the lens and source galaxy models.

# The pipeline settings we used in the advanced runners all still apply, but hyper-fitting brings with it the following
# new settings:

# - If hyper-galaxies are used to scale the noise in each component of the image (default True)

# - If the background sky is modeled throughout the pipeline (default False)

# - If the level of background noise is scaled throughout the pipeline (default True)


### PIPELINE SETTINGS ###

# When we add pipelines together, we can now define 'pipeline_settings' that dictate the behaviour of the entire
# summed pipeline. They also tag the pipeline names, to ensure that if we model the same lens with different
# pipeline settings the results on your hard-disk do not overlap.

# This means we can customize various aspects of the analysis, which will be used by all pipelines that are
# added together. In this example, our pipeline settings determine:

# - If an ExternalShear is fitted for throughout the pipeline.

# - If, after the initialize phase the light profile should be held fixed for all subsequent phases (default False)

pipeline_settings = pl.PipelineSettingsHyper(
    hyper_galaxies=True,
    hyper_image_sky=False,
    hyper_background_noise=True,
    include_shear=True,
    fix_lens_light=False,
)

### EXAMPLE ###

# So, lets do it. Below, we are going to import, add and run 3 pipelines, which do the following:

# 1) Initialize the lens and source models using a parametric source light profile.
# 2) Use this initialization to model the source as an inversion, using the lens model from the first pipeline to
#     initialize the priors.
# 3) Use this initialized source inversion to fit a more complex mass model - specifically an elliptical power-law.

from workspace.pipelines.advanced.with_lens_light.sersic.initialize import (
    lens_sersic_sie__source_sersic,
)
from workspace.pipelines.advanced.with_lens_light.sersic.power_law.from_initialize import (
    lens_sersic_power_law__source_sersic,
)

pipeline_initialize = lens_sersic_sie__source_sersic.make_pipeline(
    pipeline_settings=pipeline_settings, phase_folders=[data_type, data_name]
)

pipeline_power_law = lens_sersic_power_law__source_sersic.make_pipeline(
    pipeline_settings=pipeline_settings, phase_folders=[data_type, data_name]
)

pipeline = pipeline_initialize + pipeline_power_law

pipeline.run(data=ccd_data)
