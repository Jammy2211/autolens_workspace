"""
GUI Preprocessing: Scaled Dataset
=================================

This tool allows one to mask a bespoke noise-map for a given image of a strong lens, using a GUI.

This noise-map is primarily used for increasing the variances of pixels that have non-modeled components in an image,
for example intervening line-of-sight galaxies that are near the lens, but not directly interfering with the
analysis of the lens and source galaxies.

This GUI is adapted from the following code: https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt
import scribbler
import numpy as np

"""
Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/no_lens_light/mass_sie__source_sersic__intervening_objects`.
"""
dataset_name = "mass_sie__source_sersic__intervening_objects"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

"""
If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
"""
pixel_scales = 0.1

"""
First, load the `Imaging` dataset, so that the location of galaxies is clear when scaling the noise-map.
"""
image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"), pixel_scales=pixel_scales
)

cmap = aplt.Cmap(
    norm="log", vmin=1.0e-4, vmax=0.4 * np.max(image), linthresh=0.05, linscale=0.1
)

scribbler = scribbler.Scribbler(image=image.native, cmap=cmap)
mask = scribbler.show_mask()
mask = al.Mask2D.manual(mask=mask, pixel_scales=pixel_scales)

"""
Here, we change the image flux values to zeros. If included, we add some random Gaussian noise to most close resemble
noise in the image.
"""
background_level = al.preprocess.background_noise_map_from_edges_of_image(
    image=image, no_edges=2
)[0]

# gaussian_sigma = None
gaussian_sigma = 0.1

image = np.where(mask, 0.0, image.native)
image = al.Array2D.manual_native(array=image, pixel_scales=pixel_scales)

if gaussian_sigma is not None:
    random_noise = np.random.normal(
        loc=background_level, scale=gaussian_sigma, size=image.shape_native
    )
    image = np.where(mask, random_noise, image.native)
    image = al.Array2D.manual_native(array=image, pixel_scales=pixel_scales)

"""
The new image is plotted for inspection.
"""
array_plotter = aplt.Array2DPlotter(array=image)
array_plotter.figure_2d()

"""
Now we`re happy with the image, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""
image.output_to_fits(
    file_path=path.join(dataset_path, "image_scaled.fits"), overwrite=True
)

"""
Next, load the `Imaging` noise-map, which we will use the scale the noise-map.
"""
noise_map = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "noise_map.fits"), pixel_scales=pixel_scales
)

"""
Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
"""
array_plotter = aplt.Array2DPlotter(array=image / noise_map)
array_plotter.figure_2d()

"""
Manually increase the noise values to extremely large values, such that the analysis essentially omits them.
"""
noise_map = np.where(mask, 1.0e8, noise_map.native)
noise_map = al.Array2D.manual_native(array=noise_map, pixel_scales=pixel_scales)

"""
The signal to noise-map is the best way to determine if these regions are appropriately masked out.
"""
array_plotter = aplt.Array2DPlotter(array=image / noise_map)
array_plotter.figure_2d()

"""
Now we`re happy with the noise-map, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""
noise_map.output_to_fits(
    file_path=path.join(dataset_path, "noise_map_scaled.fits"), overwrite=True
)

"""
Lets also output the mask.
"""
mask.output_to_fits(
    file_path=path.join(dataset_path, "mask_scaled.fits"), overwrite=True
)
