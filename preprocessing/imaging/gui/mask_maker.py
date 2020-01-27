import autofit as af
import autolens as al
import autolens.plot as aplt
from preprocessing.imaging.gui import scribbler
import numpy as np

import os

# This tool allows one to mask a bespoke mask for a given image of a strong lens using an interactive GUI. This mask
# can then be loaded before a pipeline is run and passed to that pipeline so as to become the default masked used by a
# phase (if a mask function is not passed to that phase).

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

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

# First, load the imaging dataset, so that the mask can be plotted over the strong lens image.
image = al.array.from_fits(
    file_path=dataset_path + "image.fits", pixel_scales=pixel_scales
)

scribbler = scribbler.Scribbler(image=image.in_2d)
mask = scribbler.show_mask()
mask = al.mask.manual(mask_2d=np.invert(mask), pixel_scales=pixel_scales)

# Now lets plotters the image and mask, so we can check that the mask includes the regions of the image we want.
aplt.array(array=image, mask=mask)

# Now we're happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
# file in our pipelines!
mask.output_to_fits(file_path=dataset_path + "mask.fits", overwrite=True)
