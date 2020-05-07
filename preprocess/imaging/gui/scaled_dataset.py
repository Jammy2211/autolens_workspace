import autofit as af
import autolens as al
import autolens.plot as aplt
from preprocess.imaging.gui import scribbler
import numpy as np

import os

# This tool allows one to mask a bespoke noise map for a given image of a strong lens, using a GUI.

# This noise map is primarily used for increasing the variances of pixels that have non-modeled components in an image,
# for example intervening line-of-sight galaxies that are near the lens, but not directly interfering with the
# analysis of the lens and source galaxies.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../..".format(os.path.dirname(os.path.realpath(__file__)))

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the mask is stored in, e.g,
# the mask will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/mask.fits'.
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic__intervening_objects"

# Create the path where the noise map will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic_intervening_objects/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
pixel_scales = 0.1

# First, load the imaging dataset, so that the location of galaxies is clear when scaling the noise map.
image = al.Array.from_fits(
    file_path=f"{dataset_path}/image.fits", pixel_scales=pixel_scales
)

cmap = aplt.ColorMap(
    norm="log",
    norm_min=1.0e-4,
    norm_max=0.4 * np.max(image),
    linthresh=0.05,
    linscale=0.1,
)

scribbler = scribbler.Scribbler(image=image.in_2d, cmap=cmap)
mask = scribbler.show_mask()
mask = al.Mask.manual(mask_2d=mask, pixel_scales=pixel_scales)

# Here, we change the image flux values to zeros. If included, we add some random Gaussian noise to most close resemle
# noise in the image.
background_level = al.preprocess.background_noise_map_from_edges_of_image(
    image=image, no_edges=2
)[0]

# gaussian_sigma = None
gaussian_sigma = 0.1

image = np.where(mask, 0.0, image.in_2d)
image = al.Array.manual_2d(array=image, pixel_scales=pixel_scales)

if gaussian_sigma is not None:
    random_noise = np.random.normal(
        loc=background_level, scale=gaussian_sigma, size=image.shape_2d
    )
    image = np.where(mask, random_noise, image.in_2d)
    image = al.Array.manual_2d(array=image, pixel_scales=pixel_scales)

# The new image is plotted for inspection.
aplt.Array(array=image)

# Now we're happy with the image, lets output it to the dataset folder of the lens, so that we can load it from a .fits
# file in our pipelines!
image.output_to_fits(file_path=dataset_path + "image_scaled.fits", overwrite=True)

# Next, load the imaging noise map, which we will use the scale the noise map.
noise_map = al.Array.from_fits(
    file_path=f"{dataset_path}/noise_map.fits", pixel_scales=pixel_scales
)

# Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
# data_plotters.plot_signal_to_noise_map(signal_to_noise_map=image / noise_map)

# Here, we manually increase the noise values to extremely large values, such that the analysis essentially omits them.
noise_map = np.where(mask, 1.0e8, noise_map.in_2d)
noise_map = al.Array.manual_2d(array=noise_map, pixel_scales=pixel_scales)

# The signal to noise map is the best way to determine if these regions are appropriately masked out.
aplt.Array(array=image / noise_map)

# Now we're happy with the noise map, lets output it to the dataset folder of the lens, so that we can load it from a .fits
# file in our pipelines!
noise_map.output_to_fits(
    file_path=dataset_path + "noise_map_scaled.fits", overwrite=True
)

# Lets also output the mask.
mask.output_to_fits(file_path=dataset_path + "mask_scaled.fits", overwrite=True)
