import os

# This pipeline runner demonstrates how to use the position thresholding in pipelines. Checkout the pipeline
# 'autolens_workspace/pipelines/beginner/features/position_thresholding.py' for a description binning up.

# I'll assume that you are familiar with how the beginner runners work, so if any code doesn't make sense familiarize
# yourself with those first!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

config_path = workspace_path + "test/config"

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

imaging = al.Imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# We need to load the positions from a .dat file, in the same fashion as the imaging above. To draw positions
# for an image, checkout the files
# 'autolens_workspace/tools/data_making/positions_maker.py'

# The autolens_workspace comes with positions already for this dataset already. If you look in
# autolens_workspace/dataset/imaging/lens_sie__source_sersic/ you'll see a positions file!
positions = al.Coordinates.from_file(file_path=dataset_path + "positions.dat")

# When we plot the imaging dataset, we can:
# - Pass the positions to show them on the image.
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask, positions=positions)

# Finally, we import and make the pipeline as described in the runner.py file, but pass the positions into the
# 'pipeline.run() function.

from pipelines.features import position_thresholding

pipeline = position_thresholding.make_pipeline(
    phase_folders=[dataset_label, dataset_name]
)

pipeline.run(dataset=imaging, mask=mask, positions=positions)
