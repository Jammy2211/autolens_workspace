import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# This tool allows one to mask a bespoke noise-map for a given image of a strong lens.

# This noise-map is primarily used for increasing the variances of pixels that have non-modeled components in an image,
# for example intervening line-of-sight galaxies that are near the lens, but not directly interfering with the
# analysis of the lens and source galaxies.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the mask is stored in, e.g,
# the mask will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/mask.fits'.
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic__intervening_objects"

# Create the path where the noise-map will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic_intervening_objects/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
pixel_scales = 0.1

# First, load the imaging dataset, so that the location of galaxies is clear when scaling the noise-map.
image = al.array.from_fits(
    file_path=dataset_path + "image.fits", pixel_scales=pixel_scales
)

# Next, load the imaging noise-map, which we will use the scale the noise-map.
noise_map = al.array.from_fits(
    file_path=dataset_path + "noise_map.fits", pixel_scales=pixel_scales
)

# Now lets plotters the image and mask, so we can check that the mask includes the regions of the image we want.
# data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image / noise_map)

# Here, we manually increase the noise values to extremely large values, such that the analysis essentially omits them.
noise_map = noise_map.in_2d
noise_map[25:55, 77:96] = 1.0e8
noise_map[55:85, 3:27] = 1.0e8

# The signal to noise map is the best way to determine if these regions are appropriately masked out.
aplt.array(array=image / noise_map.in_1d)

# Now we're happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
# file in our pipelines!
noise_map.output_to_fits(
    file_path=dataset_path + "noise_map_scaled.fits", overwrite=True
)
