import autofit as af
from autolens.data import ccd
from autolens.data.array import mask as msk
from autolens.data.plotters import data_plotters

import os

# This tool allows one to mask a bespoke mask for a given image of a strong lens, which can then be loaded before a
# pipeline is run and passed to that pipeline so as to become the default masked used by a phase (if a mask
# function is not passed to that phase).

# Setup the path to the workspace, using a relative directory name.
workspace_path = '{}/../../../'.format(os.path.dirname(os.path.realpath(__file__)))

# The 'data name' is the name of the data folder and 'data_name' the folder the mask is stored in, e.g,
# the mask will be output as '/workspace/data/data_type/data_name/mask.fits'.
data_type = 'example'
data_name = 'lens_light_mass_and_x1_source'

# Create the path where the mask will be output, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

# If you use this tool for your own data, you *must* double check this pixel scale is correct!
pixel_scale = 0.1

# First, load the CCD imaging data, so that the mask can be plotted over the strong lens image.
image = ccd.load_image(
    image_path=data_path + 'image.fits', image_hdu=0,
    pixel_scale=pixel_scale)

# Now, create a mask for this data, using the mask function's we're used to. I'll use a circular-annular mask here,
# but I've commented over options you might want to use (feel free to experiment!)

mask = msk.Mask.circular_annular(
    shape=image.shape, pixel_scale=image.pixel_scale,
    inner_radius_arcsec=0.5, outer_radius_arcsec=2.5, centre=(0.0, 0.0))

# Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
data_plotters.plot_image(
    image=image, mask=mask)

# Now we're happy with the mask, lets output it to the data folder of the lens, so that we can load it from a .fits
# file in our pipelines!
msk.output_mask_to_fits(
    mask=mask, mask_path=data_path + 'mask.fits', overwrite=True)