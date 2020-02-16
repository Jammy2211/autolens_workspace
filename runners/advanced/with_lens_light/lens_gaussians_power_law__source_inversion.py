import os

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

### PATH STRUCTURE ####

# Advanced pipelines allow us to use the results of earlier pipelines to initialize fits to later pipelines. However,
# we have to be very careful that our runs do not write to the the same set of non-linear outputs (e.g. MultiNest
# output files). This is especially true given that we can customize so many aspects of a pipeline (the source
# pixelization / reguarization, lens light model, whether we use a shear, etc.).

# Thus, to make sure that every advanced pipeline writes results to a unique path the pipeline tags for the source,
# light and mass we previously added together in intermediate pipelines now instead create their own folder.

# For example, if an intermediate pipeline previous wrote to the path:

# 'output/intermediate/imaging/lens_name/pipeline_name/general_tag/source_tag+light_tag+mass_tag/phase_name/phase_tag

# An advanced pipeline will write to the path:

# 'output/advanced/imaging/lens_name/pipeline_name/general_tag/source_tag/light_tag/mass_tag/phase_name/phase_tag

### THIS RUNNER ###

# Using two source pipelines, a light pipeline and a mass pipeline we will fit a power-law mass model and source using
# a pixelized inversion.

# We'll use the example pipelines:
# 'autolens_workspace/pipelines/advanced/with_lens_light/source/parametric/lens_bulge_disk_sie__source_sersic.py'.
# 'autolens_workspace/pipelines/advanced/with_lens_light/source/inversion/from_parametric/lens_light_sie__source_inversion.py'.
# 'autolens_workspace/pipelines/advanced/with_lens_light/light/bulge_disk/lens_bulge_disk_sie__source.py'.
# 'autolens_workspace/pipelines/advanced/with_lens_light/mass/power_law/lens_light_power_law__source.py'.

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
dataset_label = "imaging"
dataset_name = "lens_gaussians_x3_sie__source_sersic"
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

# Advanced pipelines use the same 'Source', 'Light' and 'Mass' setup objects we used in beginner and intermediate
# pipelines. However, there are many additional options now available with these setup objects, that did not work
# for beginner and intermediate pipelines. For an explanation, checkout:

# - 'autolens_workspace/runners/advanced/doc_setup'

# The setup of earlier pipelines inform the model fitted in later pipelines. For example:

# - The pixelization and regularization scheme used in the source (inversion) pipeline will be used in the light and
#   mass pipelines.
# - The alignment of the bulge-disk lens light model used in the mass pipeline.

general_setup = al.setup.General(
    hyper_galaxies=True, hyper_image_sky=False, hyper_background_noise=True
)

source_setup = al.setup.Source(
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
    lens_light_centre=(0.0, 0.0),
    lens_mass_centre=(0.0, 0.0),
    align_light_mass_centre=False,
    no_shear=False,
    fix_lens_light=True,
    number_of_gaussians=4,
)

light_setup = al.setup.Light()

mass_setup = al.setup.Mass(no_shear=False)

setup = al.setup.Setup(
    general=general_setup, source=source_setup, light=light_setup, mass=mass_setup
)

# We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

### SOURCE ###

from pipelines.advanced.with_lens_light.source.parametric import (
    lens_gaussians_sie__source_sersic,
)
from pipelines.advanced.with_lens_light.source.inversion.from_parametric import (
    lens_light_sie__source_inversion,
)

pipeline_source__parametric = lens_gaussians_sie__source_sersic.make_pipeline(
    setup=setup, phase_folders=["advanced", dataset_label, dataset_name]
)

pipeline_source__inversion = lens_light_sie__source_inversion.make_pipeline(
    setup=setup, phase_folders=["advanced", dataset_label, dataset_name]
)

### Light ###

from pipelines.advanced.with_lens_light.light.gaussians import (
    lens_gaussians_sie__source,
)


pipeline_light__gaussians = lens_gaussians_sie__source.make_pipeline(
    setup=setup, phase_folders=["advanced", dataset_label, dataset_name]
)

### MASS ###

from pipelines.advanced.with_lens_light.mass.power_law import (
    lens_light_power_law__source,
)

pipeline_mass__power_law = lens_light_power_law__source.make_pipeline(
    setup=setup, phase_folders=["advanced", dataset_label, dataset_name]
)

### PIPELINE COMPOSITION AND RUN ###

# We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
# information throughout the analysis to later phases.

pipeline = (
    pipeline_source__parametric
    + pipeline_source__inversion
    + pipeline_light__gaussians
    + pipeline_mass__power_law
)

pipeline.run(dataset=imaging, mask=mask)
