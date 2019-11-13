import os

# Welcome to the pipeline runner. This tool allows you to load strong lens dataset, and pass it to pipelines for a
# PyAutoLens analysis. To show you around, we'll load up some example dataset and run it through some of the example
# pipelines that come distributed with PyAutoLens.

# The runner is supplied as both this Python script and a Juypter notebook. Its up to you which you use - I personally
# prefer the python script as provided you keep it relatively small, its quick and easy to comment out different lens
# names and pipelines to perform different analyses. However, notebooks are a tidier way to manage visualization - so
# feel free to use notebooks. Or, use both for a bit, and decide your favourite!

# The pipeline runner is fairly self explanatory. Make sure to checkout the pipelines in the
#  autolens_workspace/pipelines/examples/ folder - they come with detailed descriptions of what they do. I hope that you'll
# expand on them for your own personal scientific needs

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

# Create the path to the dataset folder in your autolens_workspace.
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset"]
)

# It is convenient to specify the dataset type and dataset name as a string, so that if the pipeline is applied to multiple
# images we don't have to change all of the path entries in the function below.
dataset_label = "imaging"
dataset_name = (
    "lens_bulge_disk_sie__source_sersic"
)  # Example simulated image with lens light emission and a source galaxy.
pixel_scales = 0.1

# dataset_name = 'slacs1430+4105' # Example HST imaging of the SLACS strong lens slacs1430+4150.
# pixel_scales = 0.03

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=dataset_path, folder_names=[dataset_label, dataset_name]
)

# This loads the imaging dataset, as per usual.
imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

# Plot Imaging before running.
al.plot.imaging.subplot(imaging=imaging)

# Running a pipeline is easy, we simply import it from the pipelines folder and pass the lens dataset to its run function.
# Below, we'll' use a 3 phase example pipeline to fit the dataset with a parametric lens light, mass and source light
# profile. Checkout 'autolens_workspace/pipelines/examples/lens_sie__source_sersic_parametric.py' for a full description of
# the pipeline.

# The phase folders input determines the output directory structure of the pipeline, for example the input below makes
# the directory structure:
# 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/' or
# 'autolens_workspace/output/imaging/lens_sie__source_sersic/lens_sie__source_sersic_parametric/'

# For large samples of images, we can therefore easily group lenses that are from the same sample or modeled using the
# same pipeline.


### ADDING PIPELINES ###

# Okay, so in this, the advanced runner, we're going to import multiple pipelines, and add them together :O

# What does adding two pipelines together do? Well, it means that the first pipeline will run, and then once it has
# finished, the next pipeline will run, and so on. Crucially, all of the previous results of previous pipelines are
# available to later pipelines, to pass priors and results through the pipeline. What are benefits of doing this?

# - We can create generic initialization pipelines, that initialize the lens model. Pipelines with a more specific
#   model can then continue on from these more general initialization results. In fact, you may have noticed that the
#   simple pipelines you've already been running have identical phases at the beginning of them. Adding pipelines
#   together means we don't need to repeat phases.

# - Each pipeline creates its own output folder, which is shared by the later pipelines. This means we don't
#   duplicate output for the initialize pipelines.

# - If you are working with collaborators, one can use their results / initialize pipelines to continue / tweak their
#   analysis with your own pipeline.

### PIPELINE SETTINGS ###

# When we add pipelines together, we can now define 'pipeline_settings' that dictate the behaviour of the entire
# summed pipeline. They also tag the pipeline names, to ensure that if we model the same lens with different
# pipeline settings the results on your hard-disk do not overlap.

# This means we can customize various aspects of the analysis, which will be used by all pipelines that are
# added together. In this example, our pipeline settings determine:

# - If an ExternalShear is fitted for throughout the pipeline (Default True).

# - If, after the initialize phase the light profile should be held fixed for all subsequent phases (default False)

# - If the centre of the bulge and disk components of the lens's light profile are aligned  (Default False).

# - If the rotation angle of the bulge and disk components of the lens's light profile are aligned  (Default False).

# - If the axis-ratios of the bulge and disk components of the lens's light profile are aligned  (Default False).

# - If the Disk component is modeled as a Sersic profile instead of the default Exponential (Default False).

# - The Pixelization used by the inversion phases of this pipeline (Default VoronoiMagnification).

# - The Regularization scheme used by the inversion phases of this pipeline. (Default Constant)

pipeline_settings = al.PipelineSettings(
    include_shear=True,
    fix_lens_light=False,
    align_bulge_disk_centre=False,
    align_bulge_disk_phi=False,
    align_bulge_disk_axis_ratio=False,
    disk_as_sersic=False,
    pixelization=al.pix.VoronoiMagnification,
    regularization=al.reg.Constant,
)

### EXAMPLE ###

# So, lets do it. Below, we are going to import, add and run 3 pipelines, which do the following:

# 1) Initialize the lens and source models using a parametric source light profile.
# 2) Use this initialization to model the source as an inversion, using the lens model from the first pipeline to
#     initialize the priors.
# 3) Use this initialized source inversion to fit a more complex mass model - specifically an elliptical power-law.

from pipelines.advanced.with_lens_light.bulge_disk.initialize import (
    lens_bulge_disk_sie__source_sersic,
)

from pipelines.advanced.with_lens_light.bulge_disk.inversion.from_initialize import (
    lens_bulge_disk_sie__source_inversion,
)

from pipelines.advanced.with_lens_light.bulge_disk.power_law.from_inversion import (
    lens_bulge_disk_power_law__source_inversion,
)

pipeline_initialize = lens_bulge_disk_sie__source_sersic.make_pipeline(
    pipeline_settings=pipeline_settings, phase_folders=[dataset_label, dataset_name]
)

pipeline_inversion = lens_bulge_disk_sie__source_inversion.make_pipeline(
    pipeline_settings=pipeline_settings, phase_folders=[dataset_label, dataset_name]
)

pipeline_power_law = lens_bulge_disk_power_law__source_inversion.make_pipeline(
    pipeline_settings=pipeline_settings, phase_folders=[dataset_label, dataset_name]
)

pipeline = pipeline_initialize + pipeline_inversion + pipeline_power_law

pipeline.run(dataset=imaging)
