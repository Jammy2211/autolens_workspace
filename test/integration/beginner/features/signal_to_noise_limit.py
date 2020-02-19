import os

# This pipeline runner demonstrates how to use the signal-to-noise limits in pipelines. Checkout the pipeline
# 'autolens_workspace/pipelines/beginner/features/signal_to_noise_limit.py' for a description binning up.

# I'll assume that you are familiar with how the beginner runners work, so if any code doesn't make sense familiarize
# yourself with those first!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

config_path = workspace_path + "config"

af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "/test/output"
)

dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"
pixel_scales = 0.1

### AUTOLENS + DATA SETUP ###

import autolens as al
import autolens.plot as aplt

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)

# We simply import the signal-to-noise limit pipeline and pass the signal-to-noise limit we want as an input parameter
# (which for the pipeline below, is only used in phase 1).

from pipelines.beginner.features import signal_to_noise_limit

pipeline = signal_to_noise_limit.make_pipeline(
    phase_folders=[dataset_label, dataset_name], signal_to_noise_limit=20.0
)

pipeline.run(dataset=imaging, mask=mask)
