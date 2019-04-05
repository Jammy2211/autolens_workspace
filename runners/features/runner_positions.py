from autofit.tools import path_util
from autofit import conf
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.data.plotters import ccd_plotters

import os

# This pipeline runner demonstrates how to load positions for a lens from the hard-disk, and use this to resample
# inaccurate mass models. If you haven't yet, you should read the example pipeline
# 'workspace/pipelines/features/positions.py' for a description of how positions work.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific positions functionality is used, I have deleted all comments not related to that feature.


workspace_path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))
conf.instance = conf.Config(config_path=workspace_path + 'config', output_path=workspace_path + 'output')
data_type = 'example'
data_name = 'lens_light_and_x1_source'
pixel_scale = 0.1
data_path = path_util.make_and_return_path_from_path_and_folder_names(path=workspace_path,
                                                                      folder_names=['data', data_type, data_name])
ccd_data = ccd.load_ccd_data_from_fits(image_path=data_path + 'image.fits',
                                       psf_path=data_path + 'psf.fits',
                                       noise_map_path=data_path + 'noise_map.fits',
                                       pixel_scale=pixel_scale)

# Okay, we need to load the positions from a .dat file, in the same fashion as the ccd_data above. To draw positions
# for an image, checkout the files
# 'workspace/tools/data_making/positions_maker.py'

# The example autolens_workspace data comes with positions already, if you look in
# workspace/data/example/lens_light_and_x1_source/ you'll see a positions file!
positions = ccd.load_positions(positions_path=data_path + 'positions.dat')

# When we plot the ccd data, we can:
# - Pass the positions to show them on the image.
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, positions=positions)

# Finally, we import and make the pipeline as described in the runner.py file, but pass the positions into the
# 'pipeline.run() function.

from workspace.pipelines.features import position_thresholding
pipeline = position_thresholding.make_pipeline(phase_folders=[data_type, data_name])
pipeline.run(data=ccd_data, positions=positions)