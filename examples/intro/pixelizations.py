# %%
"""
__Pixelizations__

Pixelizations reconstruct the source `Galaxy`'s light on a pixel-grid. Unlike *LightProfiles*, they are able to
reconstruct the light of non-symmetric, irregular and clumpy sources.

To reconstruct the source using a *Pixelization*, we have to impose a prior on the smoothness of the reconstructed
source, called the *Regularization*. The more we regularize the source, the smoother the source reconstruction.

The process of reconstructing a `Galaxy`'s light using a *Pixelization* and *Regularization* is called an `Inversion`,
and the term `inversion` is used throughout the **PyAutoLens** example scripts to signify that their source analysis
reconstructs its light on a pixel-grid.
"""

# %%
"""Use the WORKSPACE environment variable to determine the path to the `autolens_workspace`."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""
Load the `Imaging` data that we'll reconstruct the lensed source `Galaxy`'s light of using a pixelization.

Note how complex the lensed source galaxy looks, with multiple clumps of light - this would be very difficult to 
represent using `LightProfile`'s!
"""

# %%
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "mass_sie__source_sersic_x4"
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=0.05,
)

aplt.Imaging.subplot_imaging(imaging=imaging)

# %%
"""
We are going to fit this data, so we must create *Mask2D* and *MaskedImaging* objects.
"""

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.6
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

# %%
"""
To model the source as a *Pixelization*, we simply pass it the *Pixelization* class we want to reconstruct its light
on as well as the *Regularization* scheme describing how we smooth the source. 

First, lets use a *Rectangular* `Pixelization` with resolution 40 x 40 and *Constant* regularizaton scheme with a 
relatitvely high regularization-coefficient. The higher this coefficient, the more our source reconstruction will be
smoothed.

The lens `Galaxy`'s `EllipticalIsothermal` mass model is the true model used to simulate the data.
"""

# %%
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

# %%
"""
Now that our source-galaxy has a *Pixelization* and *Regularization*, we are able to fit the data using these in the 
same way as before, by simply passing the source galaxy to a `Tracer` and using this `Tracer` to create a *FitImaging*
object.
"""

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

# %%
"""
The fit has been performed using an `Inversion` for the source galaxy. We can see this by plotting the source-plane
of the *FitImaging* using the *subplot_of_plane* plotter. Note how the bottom-right panel shows a pixelized grid.
"""
aplt.FitImaging.subplot_of_plane(fit=fit, plane_index=1)

# %%
"""
**PyAutoLens** supports many different pixel-grids. Below, we use a *VoronoiMagnification* pixelization, which defines
the source-pixel centres in the image-plane and ray traces them to the source-plane. 

The source pixel-grid is therefore adapted to the mass-model magnification pattern, placing more source-pixel in the
highly magnified regions of the source-plane.
"""

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

aplt.FitImaging.subplot_of_plane(fit=fit, plane_index=1)

# %%
"""
By inspecting the residual-map, normalized residual-map and chi-squared-map of the *FitImaging* object, we can see how
the source reconstruction accurately fits the image of the strong lens:
"""

aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
This script has given a brief overview of *Inversions* with **PyAutoLens**. However, there is a lot more to using
*Inversions* then presented here. 

In the `autolens_workspace/examples/model/inversion` folder you will find example scripts of how to fit a model to a 
strong lens using an `Inversion`. In chapters 4 and 5 of the **HowToLens** lectures we fully cover all details of  
*Inversions*, specifically:

 - How the source reconstruction determines the flux-values of the source it reconstructs.
 - The Bayesian framework employed to choose the approrpriate level of *Regularization* and avoid overfitting noise.
 - Unphysical lens model solutions that often arise when using an `Inversion`.
 - Advanced *Pixelization* and *Regularization* schemes that adapt to the source galaxy being reconstructed.
    
"""
