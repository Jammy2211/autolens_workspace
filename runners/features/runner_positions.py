import os

# This pipeline runner demonstrates how to load positions for a lens from the hard-disk, and use this to resample
# inaccurate mass models. If you haven't yet, you should read the example pipeline
# 'workspace/pipelines/features/positions.py' for a description of how positions work.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific positions functionality is used, I have deleted all comments not related to that feature.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

config_path = workspace_path + "config"

af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

data_type = "example"
data_name = "lens_sie__source_sersic"
pixel_scale = 0.1

### AUTOLENS + DATA SETUP ###

from autolens.array import mask as msk
from autolens.data.instrument import abstract_data
from autolens.data.instrument import ccd
from autolens.data.plotters import al.ccd_plotters
from autolens.pipeline import pipeline as pl

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

ccd_data = al.load_ccd_data_from_fits(
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    pixel_scale=pixel_scale,
)

# Okay, we need to load the positions from a .dat file, in the same fashion as the ccd_data above. To draw positions
# for an image, checkout the files
# 'workspace/tools/data_making/positions_maker.py'

# The example autolens_workspace instrument comes with positions already, if you look in
# workspace/data/example/lens_light_and_x1_source/ you'll see a positions file!
positions = al.load_positions(positions_path=data_path + "positions.dat")

# When we plot the ccd instrument, we can:
# - Pass the positions to show them on the image.
al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, positions=positions)

# Finally, we import and make the pipeline as described in the runner.py file, but pass the positions into the
# 'pipeline.run() function.

from autolens_workspace.pipelines.features import position_thresholding

pipeline = position_thresholding.make_pipeline(phase_folders=[data_type, data_name])

pipeline.run(data=ccd_data, positions=positions)
