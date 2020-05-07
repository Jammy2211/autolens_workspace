import os

# This pipeline runner demonstrates how to use the position thresholding in pipelines. Checkout the pipeline
# 'autolens_workspace/pipelines/beginner/features/position_thresholding.py' for a description binning up.

# I'll assume that you are familiar with how the beginner runners work, so if any code doesn't make sense familiarize
# yourself with those first!

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../..".format(os.path.dirname(os.path.realpath(__file__)))

af.conf.instance = af.conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
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

# We need to load the positions from a .dat file, which we can do by passing the path of this file to the _from_fits
# method of the imaging. in the same fashion as the imaging above.

# To draw positions for an image, checkout the files 'autolens_workspace/preprocess/imaging/p5_positions.ipynb'

# The autolens_workspace comes with positions already for this dataset already. If you look in
# autolens_workspace/dataset/imaging/lens_sie__source_sersic/ you'll see a positions file!

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    positions_path=f"{dataset_path}/positions.dat",
    pixel_scales=pixel_scales,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# When we plot the imaging dataset, we can:

# - Pass its positions to show them on the image.
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask, positions=imaging.positions)

# Finally, we import and make the pipeline as described in the runner.py file. Because our imaging dataset has
# positions and the phase has a positions threshold, position thresholding will be used by the non-linear search.

from pipelines.features import position_thresholding

pipeline = position_thresholding.make_pipeline(phase_folders=["features"])

pipeline.run(dataset=imaging, mask=mask)
