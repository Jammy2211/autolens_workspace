"""
__Preprocess 7: Lens Light Centre (Optional)__

In this tool we mark the lens light centre(s) of a strong lens(es), which can be used as fixed values for the lens
light and mass models in a pipeline.

The benefit of doing this is a reduction in the number of free parameters fitted for as well as the removal of
systematic solutions which place the lens mass model unrealistically far from its true centre. The 'advanced' pipelines
are built to use this input centres in early phases, but remove it in later phases one an accurate lens model has
been inffered.

If you create a light_centre for your dataset, you must also update your runner to use them by loading them and
passing them to the pipeline's make function. See the 'advanced' pipelines for pipelines with these centre inputs.

Lens light centres are optional, if you struggling to get PyAutoLens to infer a good model for your dataset and you
have not tried using the lens light centres as a fixed centre for your mass model I recommend that you do.
"""

# %%
"""Lets begin by importing PyAutoFit, PyAutoLens and its plotting module."""

# %%
#%matplotlib inline

import autofit as af
import autolens as al
import autolens.plot as aplt

# %%
"""
Setup the path to the autolens_workspace, using the correct path name below.
"""

# %%
import os
workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""
The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the lens light centre is stored 
in e.g, the lens light centre will be output as 
'/autolens_workspace/dataset/dataset_type/dataset_name/light_centre.dat'.
"""

# %%
dataset_type = "imaging"
dataset_label = "with_lens_light"
dataset_name = "light_sersic__mass_sie__source_sersic"

# %%
"""
Create the path where the lens light centres will be output, which in this case is
'/autolens_workspace/dataset/imaging/with_lens_light/light_sersic__mass_sie__source_sersic'
"""

# %%
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

# %%
"""
If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
"""

# %%
pixel_scales = 0.1

# %%
"""
First, load the _Imaging_ dataset, so that the lens light centres can be plotted over the strong lens image.
"""

# %%
image = al.Array.from_fits(
    file_path=f"{dataset_path}/image.fits", pixel_scales=pixel_scales
)

# %%
"""
Now, create a lens light centre, which is a Coordinate object of (y,x) values.
"""

# %%
light_centre = al.GridCoordinates([[(0.0, 0.0)]])

# %%
"""
Now lets plot the image and lens light centre, so we can check that the centre overlaps the lens light.
"""

# %%
aplt.Array(array=image, light_profile_centres=light_centre)

# %%
"""
Now we're happy with the lens light centre(s), lets output them to the dataset folder of the lens, so that we can 
load them from a .dat file in our pipelines!
"""

# %%
light_centre.output_to_file(
    file_path=f"{dataset_path}/light_centre.dat", overwrite=True
)

# %%
"""
The workspace also includes a GUI for drawing lens light centres, which can be found at 
'autolens_workspace/preprocess/imaging/gui/light_centres.py'. This tools allows you 'click' on the image where an 
image of the lensed source is, and it will use the brightest pixel within a 5x5 box of pixels to select the coordinate.
"""
