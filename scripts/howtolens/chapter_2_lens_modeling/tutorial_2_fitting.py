"""
Tutorial 8: Fitting
===================

Up to now, we have used profiles, galaxies and tracers to create images of a strong lens. However, this is the opposite
of what most Astronomers do: normally, an Astronomer has observed an image of a strong lens, and their goal is to
determine the profiles that best represent the mass distribution of the lens galaxy and source light distribution of
the source galaxy.

To do this, we need to fit the data and determine which light and mass profiles best represent the image it contains.
We'll demonstrate how to do this using the imaging data we simulated in the previous tutorial. By comparing the images
that come out of a tracer with the data, we'll compute diagnostics that tell us how good or bad a combination of light
and mass profiles represent the strong lens we observed.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

The `dataset_path` specifies where the data was output in the last tutorial, which is the directory 
`autolens_workspace/dataset/imaging/howtolens/`.
"""
dataset_path = path.join("dataset", "imaging", "howtolens")

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

"""
__Imaging Dataset__

The `Imaging` object packages all components of an imaging dataset, in particular:

 1) The image.
 2) Its noise-map.
 3) The Point Spread Function (PSF).
    
The image and noise map are stored as an `Array2D` object, whereas the PSF is a `Kernel2D`, meaning it can be used to
perform 2D convolution.
"""
print("Image:")
print(type(dataset.data))
print("Noise-Map:")
print(type(dataset.noise_map))
print("PSF:")
print(type(dataset.psf))

"""
The `ImagingPlotter` can plot all of these attributes on a single subplot:
"""
dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Mask__

To fit an image, we must first specify a `Mask2D`, which removes certain regions of the image such that they are not 
included in the fit. We therefore want to mask out the regions of the image where the lens and source galaxies are not 
visible, for example the edges.

For the image we simulated a 3" circular mask will do the job.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

print(mask)  # 1 = True, which means the pixel is masked. Edge pixels are indeed masked.
print(mask[48:53, 48:53])  # Whereas central pixels are `False` and therefore unmasked.

"""
We can use an `ImagingPlotter` to compare the mask and the image, which is useful if we want to `tailor` a mask to 
the lensed source's light (in this example, we do not do this, but there are examples of how to do this throughout
the `autolens_workspace`).

However, the mask is not currently an attribute of the imaging, therefore we cannot make a plot of the imaging
data using the mask.

To manually plot an object over the figure of another object, we can use the `Visuals2D` object, which we used in a 
previous tutorial to plot certain pixels on an image and source plane. The `Visuals2D` object can be used to customize 
the appearance of *any* figure in **PyAutoLens** and is therefore a powerful means by which to create custom visuals!
"""
visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(data=True)

"""
Before we can fit the imaging data we need to apply the mask to it, which is done using the `apply_mask()` method. 

In addition to removing the regions of the image we do not want to fit, this also creates a new grid in the imaging 
data that consists only of image-pixels that are not masked. This grid is used for performing ray-tracing calculations
when we fit the data.
"""
dataset = dataset.apply_mask(mask=mask)

"""
Now the mask is an attribute of the imaging data we can plot it using the `Include2D` object.

Because it is an attribute, the `mask` now also automatically `zooms` our plot around the masked region only. This 
means that if our image is very large, we focus-in on the lens and source galaxies.
"""
include = aplt.Include2D(mask=True)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, include_2d=include)
dataset_plotter.figures_2d(data=True)

"""
By printing its attributes, we can see that the imaging contains everything we need to fit: a mask, the masked image, 
masked noise-map and psf.
"""
print("Mask2D")
print(dataset.mask)
print()
print("Masked Image:")
print(dataset.data)
print()
print("Masked Noise-Map:")
print(dataset.noise_map)
print()
print("PSF:")
print(dataset.psf)
print()

"""
__Masked Data Structures__

This image and noise-map again have `native` and `slim` representations. However, the `slim` representation now takes
on a slightly different meaning, it only contains image-pixels that were not masked. This can be seen by printing
the `shape_slim` attribute of the image, and comparing it to the `pixels_in_mask` of the mask.
"""
print("The number of unmasked pixels")
print(dataset.data.shape_slim)
print(dataset.noise_map.shape_slim)
print(dataset.data.mask.pixels_in_mask)

"""
We can use the `slim` attribute to print certain values of the image:
"""
print("First unmasked image value:")
print(dataset.data.slim[0])
print("First unmasked noise-map value:")
print(dataset.noise_map.slim[0])

"""
The `native` representation of the image `Array2D` retains the dimensions [total_y_image_pixels, total_x_image_pixels], 
however the exterior pixels have values of 0.0 indicating that they have been masked.
"""
print("Example masked pixels in the image's native representation:")
print(dataset.data.shape_native)
print(dataset.data.native[0, 0])
print(dataset.data.native[2, 2])
print("Example masked noise map values in its native representation:")
print(dataset.noise_map.native[0, 0])

"""
The masked imaging also has a `Grid2D`, where only coordinates which are not masked are included (the masked values 
in the native representation are set to [0.0. 0.0] to indicate they are masked).
"""
print("Masked imaging's grid")
print(dataset.grid.slim)
print(dataset.grid.native)

"""
__Fitting__

To fit an image, we first create a tracer. 

Lets use the same tracer that we simulated the imaging with in the previous tutorial, which will give us 
a 'perfect' fit.

Its worth noting that below, we use the masked imaging's grid to setup the tracer. This ensures that the tracer's 
image-plane image is the same resolution and alignment as our lens data's masked image and that the image is only 
created in unmasked pixels.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=dataset.grid)
tracer_plotter.figures_2d(image=True)

"""
__Fit__

To fit the image, we pass the `Imaging` and `Tracer` to a `FitImaging` object. This performs the following:

 1) Creates an image of the lens and source galaxies from the tracer using its `image_2d_from()` method.

 2) Blurs the tracer`s image with the data's PSF, ensuring the telescope optics are included in the fit. This 
 creates what is called the `model_image`.

 3) Computes the difference between this model-image and the observed image, creating the fit`s `residual_map`.

 4) Divides the residual-map by the noise-map, creating the fit`s `normalized_residual_map`.

 5) Squares every value in the normalized residual-map, creating the fit's `chi_squared_map`.

 6) Sums up these chi-squared values and converts them to a `log_likelihood`, which quantifies how good this tracer`s 
 fit to the data was (higher log_likelihood = better fit).
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
Using a `FitImagingPlotter` to inspect the fit, we can see that we get a very good fit. For example: 

 - The `model_image` looks like the observed galaxy data. 

 - The `residual_map` has values close to zero, confirming the difference between the model image and observed data 
 is small.

 - The  `normalized_residual_map` and `chi_squared_map` are also close to zero, confirming that when we concsider
 the noise in the data are fit is fitting it as well as one can expect.
 
We show this using the subplot of a `FitImagingPlotter` object. 

This subplot contains quantities you should be familiar with now, including the data, signal-to-noise, model images
of the individual galaxies in the strong lens and so on. 

The bottom row shows the normalized residuals (including a plot where they are limited to be between -1.0 and 
1.0 sigma) and the chi-squared map.
"""
include = aplt.Include2D(mask=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()

"""
The plotter also contains a log10 subplot, which shows identical figures on a log10 colormap in order to highlight
the fainter regions of the data.
"""
fit_plotter.subplot_fit_log10()

"""
We can print the fit`s attributes. As usual, we can choose whether to return the fits in slim or native format, with
the native data's edge values all zeros, as the edges were masked:
"""
print("Model-Data:")
print(fit.model_data.slim)
print(fit.model_data.native)
print()
print("Residual Maps:")
print(fit.residual_map.slim)
print(fit.residual_map.native)
print()
print("Chi-Squareds Maps:")
print(fit.chi_squared_map.slim)
print(fit.chi_squared_map.native)

"""
Of course, the central unmasked pixels have non-zero values.
"""
model_image = fit.model_data.native
print(model_image[48:53, 48:53])
print()

residual_map = fit.residual_map.native
print("Residuals Central Pixels:")
print(residual_map[48:53, 48:53])
print()

print("Chi-Squareds Central Pixels:")
chi_squared_map = fit.chi_squared_map.native
print(chi_squared_map[48:53, 48:53])

"""
The fit also gives a `log_likelihood`, which is a single-figure estimate of how good the model image fitted the 
imaging data (in unmasked pixels only!).
"""
print("Likelihood:")
print(fit.log_likelihood)

"""
__Fitting (incorrect fit)__

Previously, we used the same tracer to create and fit the image, giving an excellent fit. The residual map and 
chi-squared map showed no signs of the source-galaxy's light being left over. This solution will translate to one of 
the highest  log likelihood solutions possible.

Lets change the tracer, so that it is near the correct solution, but slightly off. Below, we slightly offset the lens 
galaxy, by 0.005"
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.005, 0.005), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
By plotting the fit, we see that residuals now appear at the location of the galaxy, increasing the chi-squared values 
(which determine our log_likelihood).
"""
fit_bad = al.FitImaging(dataset=dataset, tracer=tracer)

fit_bad_imaging_plotter = aplt.FitImagingPlotter(fit=fit_bad, include_2d=include)
fit_bad_imaging_plotter.subplot_fit()

"""
Lets compare the log likelihood to the value we computed above (which was 2967.0488):
"""
print("Previous Likelihood:")
print(fit.log_likelihood)
print("New Likelihood:")
print(fit_bad.log_likelihood)

"""
It decreases! As expected, this model is a worse fit to the data.

Lets change the tracer, one more time, to a solution nowhere near the correct one.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.005, 0.005),
        einstein_radius=1.5,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.2, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.5,
        effective_radius=0.8,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit_very_bad = al.FitImaging(dataset=dataset, tracer=tracer)

fit_very_bad_imaging_plotter = aplt.FitImagingPlotter(
    fit=fit_very_bad, include_2d=include
)
fit_very_bad_imaging_plotter.subplot_fit()

"""
Clearly, the model provides a terrible fit and this tracer is not a plausible representation of our strong lens dataset
(of course, we already knew that, given that we simulated it!)

The log likelihood drops dramatically, as expected.
"""
print("Previous Likelihoods:")
print(fit.log_likelihood)
print(fit_bad.log_likelihood)
print("New Likelihood:")
print(fit.log_likelihood)

"""
__Attribute Definitions__

Before finishing, lets define more quantitatively what the attributes of a fit are:

 - `model_data` = ``galaxy_images_convolved_with_psf`
 - `residual_map` = (`data` - `model_data`)
 - `normalized_residual_map` = (`data` - `model_data`) / `noise_map`
 - `chi_squared_map` = (`normalized_residuals`) ** 2.0 = ((`data` - `model_data`)**2.0)/(`variances`)
 - `chi_squared` = sum(`chi_squared_map`)

Thus, as we saw above, the lower the `chi_squared` value, the better the fit.

__Noise Normalization Term__

To compute a likelihood we assume the imaging data consists of independent Gaussian noise in every image pixel.

Therefore, to compute the `log_likelihood` we include a `noise_normalization` term, which consists of the sum
of the log of every noise-map value squared. 

Given the `noise_map` is fixed, this term does not change fot different fits, nevertheless we include it for 
completeness.
"""
print(fit.noise_normalization)

"""
__Log Likelihood__

The `log_likelihood` is defined as:

 - `log_likelihood = -0.5 (`chi_squared` + `noise_normalization`).

This makes sense, in that higher likelihood solutions correspond to lower chi-squared solutions (because of the
multiplication by a factor of -0.5).
"""
print(-0.5 * (fit.chi_squared + fit.noise_normalization))
print(fit.log_likelihood)

"""
__Likelihood Function__

When performing a fit we compute a value of `log_likelihood`, which is computed by calling the **PyAutoLens**
likelihood function -- a term used in statistics to describe how one fits a model to data.

This tutorial has likely given you a clear idea of how the likelihood is evaluated, however a more detailed 
step-by-step visual guide is provided at `autolens_workspace/*/imaging/log_likelihood_function/parametric.ipynb`.

I recommend you give this notebook a read through, in order to further clarify how a galaxy model is fitted to data.

__Wrap Up__

Congratulations, you`ve fitted your first strong lens! Perform the following exercises:

 1) In this example, we `knew` the correct solution, because we simulated the lens ourselves. In the real Universe, 
 we have no idea what the correct solution is. How would you go about finding the correct solution? Could you find a 
 solution that fits the data reasonable through trial and error?
"""
