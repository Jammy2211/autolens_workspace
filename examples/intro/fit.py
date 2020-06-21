# %%
"""
__Example: Fitting__

**PyAutoLens** uses *Tracer* objects to represent a strong lensing system. Now, we're going use these objects to
fit imaging data of a strong lens.

The autolens_workspace comes distributed with simulated images of strong lenses (an example of how these simulations
are made can be found in the 'simulate.py' exampe, with all simulator scripts located in 'autolens_workspac/simulators'.

We we begin by loading the strong lens dataset 'lens_sie__source_sersic' 'from .fits files:
"""

# %%
import autolens as al
import autolens.plot as aplt
import os

workspace_path = "{}/..".format(os.path.dirname(os.path.realpath(__file__)))
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"
dataset_path = f"{workspace_path}/dataset/{dataset_label}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=0.1,
)

# %%
"""
We can use the Imaging plotters to plot the image, noise-map and psf of the dataset.
"""

aplt.Imaging.image(imaging=imaging)
aplt.Imaging.noise_map(imaging=imaging)
aplt.Imaging.psf(imaging=imaging)

# %%
"""
The Imaging plotter also contains a subplot which plots all these properties simultaneously.
"""
aplt.Imaging.subplot_imaging(imaging=imaging)

# %%
"""
We now need to mask the data, so that regions where there is no signal (e.g. the edges) are omitted from the fit.
"""

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
)

# %%
"""
The MaskedImaging object combines the dataset with the mask.
 
Here, the Mask is also used to compute the *Grid* we used in the lensing.py tutorial to compute lensing calculations.
Note how the Grid has also had the mask applied to it.
"""
masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)
aplt.Grid(masked_imaging.grid)

# %%
"""
Here is what our image looks like with the mask applied, where PyAutoLens has automatically zoomed around the mask
to make the lensed source appear bigger.
"""
aplt.Imaging.image(imaging=masked_imaging)

# %%
"""
Following the lensing.py example, we can make a tracer from a collection of _LightProfile_, _MassProfile_ and *Galaxy*
objects.

The combination of *LightProfiles* and *MassProfiles* below is the same as those used to generate the lensed data-set,
thus it produces a tracer whose image looks exactly like the dataset. As discssed in the lensing.py tutorial, this
tracer can be extended to include additional _LightProfile_s's, _MassProfile_'s and *Galaxy*'s, for example if you 
wanted to fit a tracer where the lens light is included.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=(0.0, 0.111111),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# %%
"""
Following the lensing.py example, we can make a tracer from a collection of _LightProfile_, _MassProfile_ and *Galaxy*
objects. We can then use the *FitImaging* object to fit this tracer to the dataset. 

The fit performs the necessary tasks to create the model image we fit the data with, such as blurring the tracer's 
image with the imaging PSF. We can see this by comparing the tracer's image (which isn't PSF convolved) and the 
fit's model image (which is).
"""

aplt.Tracer.image(tracer=tracer, grid=masked_imaging.grid)

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
aplt.FitImaging.model_image(fit=fit)

# %%
"""
The fit creates the following:

    - The residual map: The model-image subtracted from the observed dataset's image.
    - The normalized residual map: The residual map divided by the noise-map.
    - The chi-squared map: The normalized residual map squared.

We'll plot all 3 of these, alongside a subplot containing them all.

For a good lens model where the model image and tracer are representative of the strong lens system the
residuals, normalized residuals and chi-squareds are minimized:

"""

aplt.FitImaging.residual_map(fit=fit)
aplt.FitImaging.normalized_residual_map(fit=fit)
aplt.FitImaging.chi_squared_map(fit=fit)
aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
In contrast, a bad lens model will show features in the residual-map and chi-squareds.

We can produce such an image by creating a tracer with different lens and source galaxies. In the example below, we 
change the centre of the source galaxy from (0.1, 0.1) to (0.12, 0.12), which leads to residuals appearing
in the fit.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.12, 0.12),
        elliptical_comps=(0.0, 0.111111),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# %%
"""
Lets create a new fit using this tracer and replot its residuals, normalized residuals and chi-squareds.
"""

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

aplt.FitImaging.residual_map(fit=fit)
aplt.FitImaging.normalized_residual_map(fit=fit)
aplt.FitImaging.chi_squared_map(fit=fit)
aplt.FitImaging.subplot_fit_imaging(fit=fit)
