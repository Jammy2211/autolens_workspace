"""
__Preprocess 8: - Scaled Dataset (Optional)__

In this tool we mark regions of the image that has signal in the proximity of the lens and source that may impact our
model fitting. By marking these regions we will scale the image to values near zero and the noise-map to large values
such that our model-fit ignores these regions.

Why not just mask these regions instead? The reason is because of inversions which reconstruct the lensed source's
light on a pixelized grid. Masking regions of the image removes them entirely from the fitting proceure. This means
their deflection angles are omitted and they are not traced to the source-plane, creating discontinuities in the
source _Pixelization_ which can negatively impact the _Regularization_ scheme.

However, by retaining them in the mask but simply scaling their values these discontinuities are omitted.
"""

# %%
"""Lets begin by importing PyAutoFit, PyAutoLens and its plotting module."""

# %%
#%matplotlib inline

import autofit as af
import autolens as al
import autolens.plot as aplt

import numpy as np

# %%
"""
Setup the path to the autolens_workspace, using a relative directory name.
"""

# %%
import os
workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""
The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the mask is stored in, e.g,
the mask will be output as '/autolens_workspace/dataset/dataset_type/dataset_name/mask.fits'.
"""

# %%
dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "mass_sie__source_sersic__intervening_objects"

# %%
"""
Create the path where the noise-map will be output, which in this case is
'/autolens_workspace/dataset/imaging/no_lens_light/mass_sie__source_sersic_intervening_objects/'
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
First, load the dataset image, so that the location of galaxies is clear when scaling the noise-map.
"""

# %%
image = al.Array.from_fits(
    file_path=f"{dataset_path}/image.fits", pixel_scales=pixel_scales
)

aplt.Array(array=image)

# %%
"""
Next, load the noise-map, which we will use the scale the noise-map.
"""

# %%
noise_map = al.Array.from_fits(
    file_path=f"{dataset_path}/noise_map.fits", pixel_scales=pixel_scales
)

aplt.Array(array=noise_map)

# %%
"""
Now lets plot the signal to noise-map, which will be reduced to nearly zero one we scale the noise.
"""

# %%
aplt.Array(array=image / noise_map)

# %%
"""
First, we manually define a mask corresponding to the regions of the image we will scale.
"""

# %%
mask = al.Mask.unmasked(shape_2d=image.shape_2d, pixel_scales=image.pixel_scales)
mask[25:55, 77:96] = True
mask[55:85, 3:27] = True

# %%
"""
We are going to change the image flux values to low values. Note zeros, but values consistent with the background
signa in the image, which we can estimate from the image itself.
"""

# %%
background_level = al.preprocess.background_noise_map_from_edges_of_image(
    image=image, no_edges=2
)[0]

# %%
"""
This function uses the mask to scale the appropriate regions of the image to the background level.
"""

# %%
image = np.where(mask, background_level, image.in_2d)
image = al.Array.manual_2d(array=image, pixel_scales=pixel_scales)

# %%
"""
To make our scaled image look as realistic as possible, we can optionally included some noise drawn from a Gaussian
distributon to replicate the noise-pattern in the image. This requires us to choose a gaussian_sigma value 
representative of the data, which you should choose via 'trial and error' until you get a noise pattern that is
visually hard to discern from the rest of the image.
"""

# %%
# gaussian_sigma = None
gaussian_sigma = 0.03

if gaussian_sigma is not None:
    random_noise = np.random.normal(
        loc=background_level, scale=gaussian_sigma, size=image.shape_2d
    )
    image = np.where(mask, random_noise, image.in_2d)
    image = al.Array.manual_2d(array=image, pixel_scales=pixel_scales)

# %%
"""
The new image is plotted for inspection.
"""

# %%
aplt.Array(array=image)

# %%
"""
Now we're happy with the image, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""

# %%
image.output_to_fits(file_path=f"{dataset_path}/image_scaled.fits", overwrite=True)

# %%
"""
Here, we manually increase the noise values at these points in the mask to extremely large values, such that the 
analysis essentially omits them.
"""

# %%
noise_map = noise_map.in_2d
noise_map[mask == True] = 1.0e8

# %%
"""
The noise-map and signal to noise-map show the noise-map being scaled in the correct regions of the image.
"""

# %%
aplt.Array(array=noise_map)
aplt.Array(array=image / noise_map.in_1d)

# %%
"""
Now we're happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""

# %%
noise_map.output_to_fits(
    file_path=f"{dataset_path}/noise_map_scaled.fits", overwrite=True
)

# %%
"""
Finally, we can output the scaled mask incase we need it in the future.
"""

# %%
mask.output_to_fits(file_path=f"{dataset_path}/mask_scaled.fits", overwrite=True)

# %%
"""
The workspace also includes a GUI for image and noise-map scaling, which can be found at 
'autolens_workspace/preprocess/imaging/gui/scaled_dataset.py'. This tools allows you 'spray paint' on the image where 
an you want to scale, allow irregular patterns (i.e. not rectangles) to be scaled.
"""
