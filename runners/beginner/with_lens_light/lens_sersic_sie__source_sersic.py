import os

### WELCOME ###

# Welcome to the pipeline runner, which loads a strong lens dataset and analyses it using a lens modeling pipeline.

# This script uses a beginner pipeline, which omits a lot of advanced PyAutoLens functionality to make the code easier
# to understand for beginners. Once you feel confident using beginner pipelines, make sure to checkout the intermediate
# pipelines to learn how to do more advanced analyses with PyAutoLens!

# You should also checkout the 'autolens_workspace/runners/features' folder. Thse pipelines describe the different
# features that customize a pipeline analysis.

### THIS RUNNER ###

# Using a pipeline composed of three phases we will fit the lens with a Sersic light model and SIE mass model and
# source using a Sersic light profile.

# We'll use the example pipeline:
# 'autolens_workspace/pipelines/beginner/with_lens_light/lens_sersic_sie__source_sersic.py'.

# Check it out now for a detailed description of the analysis!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../..".format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al
import autolens.plot as aplt

# Specify the dataset label and name, which we use to determine the path we load the data from.
dataset_label = "imaging"
dataset_name = "lens_sersic_sie__source_sersic"

# This is the pixel-to-arcsecond conversion factor of the data, you must get this right!
pixel_scales = 0.1

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# Using the dataset path, load the data (image, noise map, PSF) as an imaging object from .fits files.
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

# Next, we create the mask we'll fit this data-set with.
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# Make a quick subplot to make sure the data looks as we expect.
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

### PIPELINE SETUP ###

# The'pipeline_setup' customize a pipeline's behaviour.

# For a parametric source there are no customization options, but we must create the source setup anyway.

source_setup = al.setup.Source()

# A single Sersic light profile also has no setup options, but we'll create its setup too.

light_setup = al.setup.Light()

# The pipeline mass setup determines whether there is no external shear in the mass model or not.

mass_setup = al.setup.Mass(no_shear=False)

# Pipeline setups 'tag' the output path of a pipeline. So, if 'no_shear' is True, the pipeline's output paths
# are 'tagged' with the string 'no_shear'.

# This means you can run the same pipeline on the same data twice (with and without shear) and the results will go
# to different output folders and thus not clash with one another!

### PIPELINE SETUP + RUN ###

# First, we group all of the setup above in a pipeline setup object.

setup = al.setup.Setup(source=source_setup, light=light_setup, mass=mass_setup)

# To run a pipeline we import it from the pipelines folder, make it and pass the lens data to its run function.

# The 'phase_folders' below specify the path the pipeliine results are written to. Our output will go to the path
# 'autolens_workspace/output/beginner/dataset_label/dataset_name/' or equivalently
# 'autolens_workspace/output/beginner/imaging/lens_sersic_sie__source_sersic/'

from pipelines.beginner.with_lens_light import lens_sersic_sie__source_sersic

pipeline = lens_sersic_sie__source_sersic.make_pipeline(
    setup=setup, phase_folders=["beginner", dataset_label, dataset_name]
)

pipeline.run(dataset=imaging, mask=mask)
