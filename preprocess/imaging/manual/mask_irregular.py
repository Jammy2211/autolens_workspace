import autolens as al
import autolens.plot as aplt
import numpy as np

# %%
"""
This tool allows one to mask a bespoke mask for a given image of a strong lens, which can then be loaded
before a pipeline is run and passed to that pipeline so as to become the default masked used by a phase (if a mask
function is not passed to that phase).

This tool creates an irregular mask, which can form any shape and is not restricted to circles, annuli, ellipses,
etc. This mask is created as follows:

1) Blur the observed image with a Gaussian kernel of specified FWHM.
2) Compute the absolute S/N map of that blurred image and the noise-map.
3) Create the mask for all pixels with a S/N above a theshold value.

For strong lenses without a lens light component this masks create a source-only mask. If the lens light is included
it includes the lens light and source.

The following parameters determine the behaviour of this function:
"""

# %%
"""
The sigma value (e.g. FWHM) of the Gaussian the image is blurred with and the S/N threshold defining above which a 
image-pixel value must be to not be masked.
"""

# %%
blurring_gaussian_sigma = 0.1
signal_to_noise_threshold = 10.0

# %%
"""
Next, lets setup the path to our current working directory. I recommend you use the 'autolens_workspace' directory 
and place your dataset in the 'autolens_workspace/preprocess/imaging/data_raw' directory.

For this tutorial, we'll use the 'autolens_workspace/preprocess/data_raw/imaging' directory. The folder 'data_raw' 
contains example data we'll use in this tutorial.
"""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

dataset_path = f"{workspace_path}/preprocess/imaging/data_raw/imaging"

image = al.Array.from_fits(file_path=f"{dataset_path}/image.fits", pixel_scales=0.1)
noise_map = al.Array.from_fits(
    file_path=f"{dataset_path}/noise_map.fits", pixel_scales=0.1
)

# %%
"""
Create the 2D Gaussian that the image is blurred with. This blurring smooths over noise in the image, which will 
otherwise lead unmasked values with in individual pixels if not smoothed over correctly.
"""

# %%
blurring_gaussian = al.Kernel.from_gaussian(
    shape_2d=(31, 31), pixel_scales=image.pixel_scales, sigma=blurring_gaussian_sigma
)

# %%
"""
Blur the image with this Gaussian smoothing kernel and plot the resulting image.
"""

# %%
blurred_image = blurring_gaussian.convolved_array_from_array(array=image)
aplt.Array(array=blurred_image)

# %%
"""Now compute the absolute signal-to-noise map of this blurred image, given the noise-map of the observed dataset."""

# %%
blurred_signal_to_noise_map = blurred_image / noise_map
aplt.Array(array=blurred_signal_to_noise_map)

# %%
"""Now create the mask in 2ll pixels where the signal to noise is above some threshold value."""

# %%
mask = np.where(
    blurred_signal_to_noise_map.in_2d > signal_to_noise_threshold, False, True
)
mask = al.Mask.manual(mask=mask, pixel_scales=image.pixel_scales, sub_size=1)
aplt.Array(array=image, mask=mask)

# %%
"""
Now we're happy with the mask, lets output it to the dataset folder of the lens, so that we can load it from a .fits
file in our pipelines!
"""

# %%
mask.output_to_fits(file_path=dataset_path + "mask.fits", overwrite=True)
