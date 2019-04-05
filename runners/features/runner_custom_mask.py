from autofit.tools import path_util
from autofit import conf
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.data.plotters import ccd_plotters

import os

# This pipeline runner demonstrates how to load a custom mask for a lens from the hard-disk, and use this as the
# default mask in a pipeline. To be clear, the mask used in a pipeline is descided as follows:

# - If a phase is NOT supplied with a mask_function, the custom input mask is used.
# - If a phase IS supplied with a mask_function, the mask described by the mask_function is used instead.
# - Regardless of the mask used above, use of the inner_circular_mask_radii function will add that circular mask
#   to the mask.

# Most of this runner repeats the command described in the 'runner.'py' file. Therefore, to make it clear where the
# specific mask functionality is used, I have deleted all comments not related to that feature.

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

# Okay, we need to load the mask from a .fits file, in the same fashion as the ccd_data above. To draw a mask for an
# image, checkout the files
# 'workspace/tools/data_making/mask_maker.py and workspace/tools/data_making/mask_maker_irregular.py'

# The example autolens_workspace data comes with a mask already, if you look in
# workspace/data/example/lens_light_and_x1_source/ you'll see a mask.fits file!
mask = msk.load_mask_from_fits(mask_path=data_path + 'mask.fits', pixel_scale=pixel_scale)

# When we plot the ccd data, we can:
# - Pass the mask to show it on the image.
# - Extract only the regions of the image in the mask, to remove contaminating bright sources away from the lens.
# - zoom in around the mask to emphasize the lens.
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask, extract_array_from_mask=True, zoom_around_mask=True)

# Finally, we import and make the pipeline as described in the runner.py file, but pass the mask into the
# 'pipeline.run() function.

from workspace.pipelines.examples import lens_sersic_sie_source_x1_sersic
pipeline = lens_sersic_sie_source_x1_sersic.make_pipeline(phase_folders=[data_type, data_name])
pipeline.run(data=ccd_data, mask=mask)