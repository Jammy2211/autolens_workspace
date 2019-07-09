import autofit as af
from autolens.data import ccd
from autolens.data.plotters import ccd_plotters
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# Welcome to the hyper pipeline runner! This script is identical to the
# 'runners/simple/runner_lens_light_mass_and_source.py' script, except at the end when we add pipelines together. So,
# if you already know how the simple runners work, jump ahead to our pipeline imports. If you don't, I recommmend you
# checout the 'simple' pipelines, before using this script.

### The code between the dashed ---- lines is identical to 'runners/simple/runner_lens_light_mass_and_source.py' ###

# ----------------------------------------------------------------------------------------------------------

# Welcome to the pipeline runner. This tool allows you to load strong lens data, and pass it to pipelines for a
# PyAutoLens analysis. To show you around, we'll load up some example data and run it through some of the example
# pipelines that come distributed with PyAutoLens.

# The runner is supplied as both this Python script and a Juypter notebook. Its up to you which you use - I personally
# prefer the python script as provided you keep it relatively small, its quick and easy to comment out different lens
# names and pipelines to perform different analyses. However, notebooks are a tidier way to manage visualization - so
# feel free to use notebooks. Or, use both for a bit, and decide your favourite!

# The pipeline runner is fairly self explanatory. Make sure to checkout the pipelines in the
#  workspace/pipelines/examples/ folder - they come with detailed descriptions of what they do. I hope that you'll
# expand on them for your own personal scientific needs

# Setup the path to the workspace, using a relative directory name.
workspace_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + 'config', output_path=workspace_path + 'output')

# Create the path to the data folder in your workspace.
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data'])

# It is convenient to specify the data type and data name as a string, so that if the pipeline is applied to multiple
# images we don't have to change all of the path entries in the load_ccd_data_from_fits function below.
data_type = 'example'
data_name = 'lens_mass_and_x1_source' # Example simulated image with lens light emission and a source galaxy.
pixel_scale = 0.1

# data_name = 'slacs1430+4105' # Example HST imaging of the SLACS strong lens slacs1430+4150.
# pixel_scale = 0.03

# Create the path where the data will be loaded from, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=data_path, folder_names=[data_type, data_name])

# This loads the CCD imaging data, as per usual.
ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + 'image.fits',
    psf_path=data_path + 'psf.fits',
    noise_map_path=data_path + 'noise_map.fits',
    pixel_scale=pixel_scale)

# Plot CCD before running.
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# Running a pipeline is easy, we simply import it from the pipelines folder and pass the lens data to its run function.
# Below, we'll' use a 3 phase example pipeline to fit the data with a parametric lens light, mass and source light
# profile. Checkout 'workspace/pipelines/examples/lens_light_and_x1_source_parametric.py' for a full description of
# the pipeline.

# The phase folders input determines the output directory structure of the pipeline, for example the input below makes
# the directory structure:
# 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/' or
# 'autolens_workspace/output/example/lens_light_and_x1_source/lens_light_and_x1_source_parametric/'

# For large samples of images, we can therefore easily group lenses that are from the same sample or modeled using the
# same pipeline.

# --------------------------------------------------------------------------------------------------------

# Okay, so in this, the hyper runner, we're going to use a hyper inversion. This is an invversion where:

# - The source pixelization adapts to the surface brightness of the reconstructed source.
# - The source regularization is adaptive, reducing regularization in the source's brightest regions.

# Using a hyper inversion is easy. We just pass them to our inversion pipeline and those after. In this runner, we'll
# turn off all other hyper galaxy function, that is discussed in other hyper runners (hyper galaxies, background).

from workspace.pipelines.hyper.no_lens_light.initialize import lens_sie_shear_source_sersic
from workspace.pipelines.hyper.no_lens_light.inversion.from_initialize import lens_sie_shear_source_inversion
from workspace.pipelines.hyper.no_lens_light.power_law.from_inversion import lens_pl_shear_source_inversion

pl_pixelization = pix.VoronoiBrightnessImage
pl_regularization = reg.AdaptiveBrightness

pipeline_initialize = lens_sie_shear_source_sersic.make_pipeline(
    pl_hyper_galaxies=False, pl_hyper_background_sky=False, pl_hyper_background_noise=False,
    phase_folders=[data_type, data_name])

pipeline_inversion = lens_sie_shear_source_inversion.make_pipeline(
    pl_hyper_galaxies=False, pl_hyper_background_sky=False, pl_hyper_background_noise=False,
    pl_pixelization=pl_pixelization, pl_regularization=pl_regularization,
    phase_folders=[data_type, data_name],
    positions_threshold=1.0)

pipeline_power_law = lens_pl_shear_source_inversion.make_pipeline(
    pl_hyper_galaxies=False, pl_hyper_background_sky=False, pl_hyper_background_noise=False,
    pl_pixelization=pl_pixelization, pl_regularization=pl_regularization,
    phase_folders=[data_type, data_name],
    positions_threshold=1.0)

pipeline = pipeline_initialize + pipeline_inversion + pipeline_power_law

pipeline.run(data=ccd_data)