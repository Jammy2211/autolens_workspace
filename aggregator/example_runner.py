# Welcome to the example aggregator pipeline runner. This is a hyper runner, which analyses two example images to
# setup the pipeline results on your hard-disk to demonstrate the aggregator. You should run this now, which should
# take ~ 10 minutes

# The rest of the text is the usual runner script you are used to, so you should be good to checkout the
# 'aggregator.ipyn' notebook.

######################################

import autofit as af
from autolens.data.instrument import abstract_data
from autolens.data.instrument import ccd
from autolens.data.array import mask as msk

import os

# Setup the path to the workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

# It is convenient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the load_ccd_data_from_fits function below.

data_type = "example"
data_name = (
    "lens_sersic_sie__source_sersic"
)  # An example simulated image without any lens light and a source galaxy.
pixel_scale = 0.1

# Create the path where the instrument will be loaded from, which in this case is
# '/workspace/data/example/lens_light_mass_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    pixel_scale=pixel_scale,
)

mask = msk.load_mask_from_fits(
    mask_path=data_path + "mask.fits", pixel_scale=pixel_scale
)

# Running a pipeline is easy, we simply import it from the pipelines folder and pass the lens instrument to its run function.
# Below, we'll use a 3 phase example pipeline to fit the instrument with a mass model and pixelized source reconstruction.
# Checkout _workspace/pipelines/examples/lens_sie__source_inversion.py' for a full description of
# the pipeline.

from workspace.pipelines.simple import lens_sersic_sie__source_sersic

pipeline = lens_sersic_sie__source_sersic.make_pipeline(
    phase_folders=[data_type, data_name]
)

pipeline.run(data=ccd_data, mask=mask)

data_type = "example"
data_name = "lens_sersic_sie__source_sersic__2"
pixel_scale = 0.1

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    pixel_scale=pixel_scale,
)

mask = msk.load_mask_from_fits(
    mask_path=data_path + "mask.fits", pixel_scale=pixel_scale
)

from workspace.pipelines.simple import lens_sersic_sie__source_sersic

pipeline = lens_sersic_sie__source_sersic.make_pipeline(
    phase_folders=[data_type, data_name]
)

pipeline.run(data=ccd_data, mask=mask)
