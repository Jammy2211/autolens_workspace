"""
GUI Preprocessing: Mask
=======================

This tool allows one to mask a bespoke mask for a given image of a strong lens using an interactive GUI. This mask
can then be loaded before a pipeline is run and passed to that pipeline so as to become the default masked used by a
search (if a mask function is not passed to that search).

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
folder `dataset/imaging/with_lens_light/mass_sie__source_sersic`.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

"""
If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
"""
pixel_scales = 0.1

"""
First, load the `Imaging` dataset, so that the mask can be plotted over the strong lens image.
"""
image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "image.fits"), pixel_scales=pixel_scales
)

"""
Load the GUI for drawing the mask. Push Esc when you are finished drawing the mask.
"""
scribbler = scribbler.Scribbler(image=image.native)
mask = scribbler.show_mask()
mask = al.Mask2D.manual(mask=np.invert(mask), pixel_scales=pixel_scales)

"""
Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
"""
visuals_2d = aplt.Visuals2D(mask=mask)
array_2d_plotter = aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)
array_2d_plotter.figure_2d()

"""
Now we`re happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""
mask.output_to_fits(file_path=path.join(dataset_path, "mask.fits"), overwrite=True)
