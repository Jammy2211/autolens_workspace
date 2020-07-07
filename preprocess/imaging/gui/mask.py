import autofit as af
import autolens as al
import autolens.plot as aplt
from preprocess.imaging.gui import scribbler
import numpy as np

# This tool allows one to mask a bespoke mask for a given image of a strong lens using an interactive GUI. This mask
# can then be loaded before a pipeline is run and passed to that pipeline so as to become the default masked used by a
# phase (if a mask function is not passed to that phase).

# %%
"""Setup the path to the autolens workspace, using the project pyprojroot which determines it automatically."""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the mask is stored in, e.g,
# the mask will be output as '/autolens_workspace/dataset/dataset_type/dataset_name/mask.fits'.
dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "lens_sie__source_sersic"

# Create the path where the mask will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_type, dataset_label, dataset_name]
)

# If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
pixel_scales = 0.1

# First, load the imaging dataset, so that the mask can be plotted over the strong lens image.
image = al.Array.from_fits(
    file_path=f"{dataset_path}/image.fits", pixel_scales=pixel_scales
)

scribbler = scribbler.Scribbler(image=image.in_2d)
mask = scribbler.show_mask()
mask = al.Mask.manual(mask=np.invert(mask), pixel_scales=pixel_scales)

# Now lets plot the image and mask, so we can check that the mask includes the regions of the image we want.
aplt.Array(array=image, mask=mask)

# Now we're happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
# file in our pipelines!
mask.output_to_fits(file_path=f"{dataset_path}/mask.fits", overwrite=True)
