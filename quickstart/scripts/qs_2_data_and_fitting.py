# %%
"""
__Data and Fitting__

Lets fit a strong lens image in PyAutoLens, using mass-profiles, light-profiles, galaxies and a tracer. First, we
need some data, so lets load an example image that comes prepacked with PyAutoLens which we'll load from a fits file.
"""

# %%
#%matplotlib inline

import autolens as al
import autolens.plot as aplt

# %%
"""
Change the path below to that of your workspace.
"""

# %%
workspace_path = "/path/to/user/autolens_workspace"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"

# %%
"""
The data path specifies where the data is located and loaded from.
"""

# %%
dataset_path = f"{workspace_path}/dataset/imaging/lens_sie__source_sersic"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.1,
)

# %%
"""
To fit the data, we need the following four things:

1) The image of the strong lens.
2) A noise map, which weights how much each image pixels contributes to the fit.
3) The PSF, which defines how the image is blurred during data acquisition.
4) The pixel-scale of the image defining the arcsecond to pixel conversion.
"""

# %%
aplt.Imaging.subplot_imaging(imaging=imaging)

# %%
"""
To fit an image we also specify a mask, which describes which sections of the image we fit. We mask out regions of 
the image where the lens and source galaxies are not visible, (e.g. the edges). For our image a 3" circular mask is 
ideal and we can plot this mask over our image, as well as using it to 'extract' and 'zoom-in' on the region of interest.
"""

# %%
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
Next, we create a 'MaskedImaging' object, which is a 'package' of all parts of a data-set we need to fit it:

1) The imaging-data, e.g. the image, PSF and noise map.
2) The mask.
3) A grid aligned with the image's pixels: ray-tracing uses the data's masked grid coordinates.
"""

# %%
masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

# %%
"""
We fit an image using a tracer. Below, I create a tracer with the same galaxies used to simulate the image. Our fit 
will therefore be 'perfect'. We use the masked_imaging's grid to setup the tracer, ensuring our ray-tracing fit is 
aligned with the image-data and mask.
"""

# %%
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

aplt.Tracer.profile_image(tracer=tracer, grid=masked_imaging.grid)

# %%
"""
To fit the image, we pass the masked imaging and tracer to a FitImaging object. This performs the following steps:

1) Blurs the tracer image with the lens data's PSF, ensuring that the telescope optics are accounted for by the fit. 
   This creates the fit's 'model_image'.
2) Computes the difference between this model_image and the observed image, creating the fit's 'residual_map'.
3) Divides the residuals by the noise map and squares each value, creating the fit's 'chi_squared_map'.
4) Sums up these chi-squared values and converts them to a 'log_likelihood', which quantifies how good the tracer's fit to 
   the data was (higher log_likelihood = better fit).
"""

# %%
fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))

# %%
"""
We can create a sub-plot of the fit to an individual plane in the tracer, showing:

1) The observed image (again).
2) The part of the observed image that that plane's galaxies are fitting.
3) The model image of that plane's galaxies
4) The model galaxy in the (unlensed) source plane.
"""

# %%
aplt.FitImaging.subplot_of_plane(
    fit=fit, plane_index=1, include=aplt.Include(mask=True)
)

# %%
"""
A fit also provides a log likelihood, which is a single-figure estimate of how good the model image fitted the simulated 
image (in unmasked pixels only!).
"""

# %%
print("Likelihood:")
print(fit.log_likelihood)

# %%
"""
Above, we used the same tracer to create and fit the image, giving us a 'perfect' fit where the residuals and 
chi-squareds showed no resdial features of the source galaxy's light. This solution will translate to one of the 
highest-log_likelihood solutions.

Lets change the tracer so that it's near the correct solution but slightly off, by offsetting the lens galaxy by 0.02".
"""

# %%
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.02, 0.02), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))

aplt.FitImaging.subplot_of_plane(
    fit=fit, plane_index=1, include=aplt.Include(mask=True)
)

# %%
"""
Residuals now appear at the locations the source galaxy, producing increased chi-squareds which determine our 
goodness-of-fit.

Lets compare the log likelihood to the value we computed above (which was 4440.90):
"""

# %%
print("Previous Likelihood:")
print(4440.90)
print("New Likelihood:")
print(fit.log_likelihood)

# %%
"""
It decreases! This model was a worse fit to the data.

Lens modeling in PyAutolens boils down to one simple task. Given an image of a strong lens we must find the combination 
of light and mass profiles that create a model image that matches the observed image. For real strong data we have no 
idea what these values are, and finding these values is what lens modeling is!
"""
