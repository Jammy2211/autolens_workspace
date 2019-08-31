import autofit as af
import autolens as al

import os

# This tool allows one to mask a bespoke noise-map for a given image of a strong lens, which can then be loaded before a
# pipeline is run and passed to that pipeline so as to become the default masked used by a phase (if a mask
# function is not passed to that phase).

# This noise-map is primarily used for increasing the variances of pixels that have non-modeled components in an image,
# for example intervening line-of-sight galaxies that are near the lens, but not directly interfering with the
# analysis of the lens and source galaxies.

# Setup the path to the workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# The 'data name' is the name of the data folder and 'data_name' the folder the mask is stored in, e.g,
# the mask will be output as '/workspace/data/data_type/data_name/mask.fits'.
data_type = "example"
data_name = "lens_sie__source_sersic__intervening_objects"

# Create the path where the noise-map will be output, which in this case is
# '/workspace/data/example/lens_light_and_x1_source_intervening_objects/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

# If you use this tool for your own instrument, you *must* double check this pixel scale is correct!
pixel_scale = 0.1

# First, load the CCD imaging data, so that the location of galaxies is clear when scaling the noise-map.
image = al.load_image(
    image_path=data_path + "image.fits", image_hdu=0, pixel_scale=pixel_scale
)

# Next, load the CCD imaging noise-map, which we will use the scale the noise-map.
noise_map = al.load_noise_map(
    noise_map_path=data_path + "noise_map.fits",
    noise_map_hdu=0,
    pixel_scale=pixel_scale,
)

# Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
# data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image / noise_map)

# Here, we manually increase the noise values to extremely large values, such that the analysis essentially omits them.
noise_map[25:55, 77:96] = 1.0e8
noise_map[55:85, 3:27] = 1.0e8

# The signal to noise map is the best way to determine if these regions are appropriately masked out.
al.data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image / noise_map)

# Now we're happy with the mask, lets output it to the data folder of the lens, so that we can load it from a .fits
# file in our pipelines!
al.array_util.numpy_array_2d_to_fits(
    array_2d=noise_map, file_path=data_path + "noise_map_scaled.fits", overwrite=True
)
