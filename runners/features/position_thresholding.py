import os

# This pipeline runner demonstrates how to load positions for a lens and use these to resample inaccurate mass models.
# If you haven't yet, you should read the example pipeline
# 'autolens_workspace/pipelines/features/position_thresholding.py' for a description of how positions work.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific positions functionality is used, I have deleted all comments not related to that feature.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

config_path = workspace_path + "config"

af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"
pixel_scales = 0.1

### AUTOLENS + DATA SETUP ###

import autolens as al

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)

# Okay, we need to load the positions from a .dat file, in the same fashion as the imaging above. To draw positions
# for an image, checkout the files
# 'autolens_workspace/tools/data_making/positions_maker.py'

# The example autolens_workspace dataset_label comes with positions already, if you look in
# autolens_workspace/dataset/imaging/lens_sie__source_sersic/ you'll see a positions file!
positions = al.positions.from_file(positions_path=dataset_path + "positions.dat")

# When we plotters the imaging dataset_label, we can:
# - Pass the positions to show them on the image.
al.plot.imaging.subplot(imaging=imaging, positions=positions)

# Finally, we import and make the pipeline as described in the runner.py file, but pass the positions into the
# 'pipeline.run() function.

from pipelines.features import position_thresholding

pipeline = position_thresholding.make_pipeline(
    phase_folders=[dataset_label, dataset_name]
)

pipeline.run(dataset=imaging, positions=positions)
