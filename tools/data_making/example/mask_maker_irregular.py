import autofit as af
from autolens.data import ccd
from autolens.data.array import scaled_array
from autolens.data.array import mask as msk
from autolens.plotters import array_plotters

import numpy as np
import os

# This tool allows one to mask a bespoke mask for a given image of a strong lens, which can then be loaded
# before a pipeline is run and passed to that pipeline so as to become the default masked used by a phase (if a mask
# function is not passed to that phase).

# This tool creates an irregular mask, which can form any shape and is not restricted to circles, annuli, ellipses,
# etc. This mask is created as follows:

# 1) Blur the observed image with a Gaussian kernel of specified FWHM.
# 2) Compute the absolute S/N map of that blurred image and the noise-map.
# 3) Create the mask for all pixels with a S/N above a theshold value.

# The following parameters determine the behaviour of this function:
blurring_gaussian_sigma = 0.01 # The FWHM of the Gaussian the image is blurred with
signal_to_noise_threshold = 1.0 # The threshold S/N value the blurred image must be above to not be masked.

# Setup the path to the workspace, using a relative directory name.
workspace_path = '{}/../../../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + 'config', output_path=workspace_path + 'output')

# The 'data name' is the name of the data folder and 'data_name' the folder the mask is stored in, e.g,
# the mask will be output as '/workspace/data/data_type/data_name/mask.fits'.
data_type = 'example'
data_name = 'lens_light_and_x1_source'

# Create the path where the mask will be output, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

# If you use this tool for your own data, you *must* double check this pixel scale is correct!
pixel_scale = 0.1

# First, load the CCD imaging data and noise-map, so that the mask can be plotted over the strong lens image.
image = ccd.load_image(
    image_path=data_path + 'image.fits', image_hdu=0,
    pixel_scale=pixel_scale)

noise_map = ccd.load_image(
    image_path=data_path + 'noise_map.fits', image_hdu=0,
    pixel_scale=pixel_scale)

# Create the 2D Gaussian that the image is blurred with. This blurring is an attempt to smooth over noise in the
# image, which will lead unmasked values with in individual pixels if not smoothed over correctly.
blurring_gaussian = ccd.PSF.from_gaussian(
    shape=(31, 31), pixel_scale=pixel_scale, sigma=blurring_gaussian_sigma)

blurred_image = blurring_gaussian.convolve(image)

blurred_image = scaled_array.ScaledSquarePixelArray(
    array=blurred_image, pixel_scale=pixel_scale)

array_plotters.plot_array(array=blurred_image)

# Now compute the absolute signal-to-noise map of this blurred image, given the noise-map of the observed data.
blurred_signal_to_noise_map = blurred_image / noise_map

blurred_signal_to_noise_map = scaled_array.ScaledSquarePixelArray(
    array=blurred_signal_to_noise_map, pixel_scale=pixel_scale)

array_plotters.plot_array(
    array=blurred_signal_to_noise_map)

# Now create the mask in sall pixels where the signal to noise is above some threshold value.
mask = np.where(blurred_signal_to_noise_map > signal_to_noise_threshold, False, True)

mask = msk.Mask(
    array=mask, pixel_scale=pixel_scale)

array_plotters.plot_array\
    (array=image, mask=mask)

# Now we're happy with the mask, lets output it to the data folder of the lens, so that we can load it from a .fits
# file in our pipelines!
msk.output_mask_to_fits(
    mask=mask, mask_path=data_path + 'mask.fits', overwrite=True)