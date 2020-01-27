import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np
import os

# This tool allows one to mask a bespoke mask for a given image of a strong lens, which can then be loaded
# before a pipeline is run and passed to that pipeline so as to become the default masked used by a phase (if a mask
# function is not passed to that phase).

# This tool creates an irmask, which can form any shape and is not restricted to circles, annuli, ellipses,
# etc. This mask is created as follows:

# 1) Blur the observed image with a Gaussian kernel of specified FWHM.
# 2) Compute the absolute S/N map of that blurred image and the noise-map.
# 3) Create the mask for all pixels with a S/N above a theshold value.

# The following parameters determine the behaviour of this function:
blurring_gaussian_sigma = 0.02  # The FWHM of the Gaussian the image is blurred with
signal_to_noise_threshold = (
    10.0
)  # The threshold S/N value the blurred image must be above to not be masked.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path + "config", output_path=workspace_path + "output"
)

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the mask is stored in, e.g,
# the mask will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/mask.fits'.
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"

# Create the path where the mask will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
pixel_scales = 0.1

# First, load the imaging dataset and noise-map, so that the mask can be plotted over the strong lens image.
image = al.array.from_fits(
    file_path=dataset_path + "image.fits", pixel_scales=pixel_scales
)

noise_map = al.array.from_fits(
    file_path=dataset_path + "noise_map.fits", pixel_scales=pixel_scales
)

# Create the 2D Gaussian that the image is blurred with. This blurring is an attempt to smooth over noise in the
# image, which will lead unmasked values with in individual pixels if not smoothed over correctly.
blurring_gaussian = al.kernel.from_gaussian(
    shape_2d=(31, 31), pixel_scales=pixel_scales, sigma=blurring_gaussian_sigma
)

blurred_image = blurring_gaussian.convolved_array_from_array(array=image)

aplt.array(array=blurred_image)

# Now compute the absolute signal-to-noise map of this blurred image, given the noise-map of the observed dataset.
blurred_signal_to_noise_map = blurred_image / noise_map

aplt.array(array=blurred_signal_to_noise_map)

# Now create the mask in sall pixels where the signal to noise is above some threshold value.
mask = np.where(
    blurred_signal_to_noise_map.in_2d > signal_to_noise_threshold, False, True
)

mask = al.mask.manual(mask_2d=mask, pixel_scales=pixel_scales, sub_size=1)

aplt.array(array=image, mask=mask)

# Now we're happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
# file in our pipelines!
mask.output_to_fits(file_path=dataset_path + "mask.fits", overwrite=True)
