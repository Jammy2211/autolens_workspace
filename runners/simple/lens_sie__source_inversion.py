import os

# Welcome to the pipeline runner. This tool allows you to load dataset on strong lenses, and pass it to pipelines for a
# strong lens analysis. To show you around, we'll load up some example dataset and run it through some of the example
# pipelines that come distributed with PyAutoLens, in the 'autolens_workspace/pipelines/simple' folder.
#
# In this  runner, our analysis assumes there is no lens light component that requires modeling and subtraction.

# Why is this folder called 'simple'? Well, we actually recommend you break pipelines up. Details of how to do this
# can be found in the 'autolens_workspace/runners/advanced' folder, but its a bit conceptually difficult. So, to
# familiarize yourself with PyAutoLens, I'd stick to the simple pipelines for now!

# You should also checkout the 'autolens_workspace/runners/features' folder. Here, you'll find features that are
# available in pipelines, that allow you to customize the analysis.

# This runner is supplied as both this Python script and a Juypter notebook. Its up to you which you use - I personally
# prefer the python script as provided you keep it relatively small, its quick and easy to comment out different lens
# names and pipelines to set off different analyses. However, notebooks are a tidier way to manage visualization - so
# feel free to use notebooks. Or, use both for a bit, and decide your favourite!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al

# It is convenient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the function below.

dataset_label = "slacs"
dataset_name = (
    "slacs1430+4105"
)  # An example simulated image without any lens light and a source galaxy.
pixel_scales = 0.05

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_light_mass_and_x1_source/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

al.plot.imaging.subplot(imaging=imaging)

# Running a pipeline is easy, we simply import it from the pipelines folder and pass the lens dataset to its run function.
# Below, we'll use a 3 phase example pipeline to fit the dataset with a mass model and pixelized source reconstruction.
# Checkout autolens_workspace/pipelines/examples/lens_sie__source_inversion.py' for a full description of
# the pipeline.

from pipelines.simple import lens_sie__source_inversion

pipeline = lens_sie__source_inversion.make_pipeline(
    phase_folders=[dataset_label, dataset_name], include_shear=True
)

pipeline.run(dataset=imaging)

# Another example pipeline is shown below, which fits an image generated using two source galaxies with a
# pipeline that fits two source galaxies. The source light is modeled as parametric Sersic functions.

# dataset_label = 'example'
# dataset_name = 'lens_mass__source_sersic_x2' # An example simulated image without any lens light and two source galaxies.
# pixel_scales = 0.1
#
# dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(path=workspace_path,
#                                                                       folder_names=['dataset_label', dataset_label, dataset_name])
#
# imaging = al.imaging.from_fits(image_path=dataset_path + 'image.fits',
#                                        psf_path=dataset_path + 'psf.fits',
#                                        noise_map_path=dataset_path + 'noise_map.fits',
#                                        pixel_scales=pixel_scales)
#
# al.plot.imaging.subplot(imaging=imaging)

# from autolens_workspace.pipelines.simple import lens_sie__source_x2_sersic
# pipeline = lens_sie__source_x2_sersic.make_pipeline(phase_folders=[dataset_label, dataset_name])
# pipeline.run(dataset_label=imaging)
