from autolens.data.instrument import ccd
from autolens.array import mask as ma
from autolens.lens import ray_tracing, lens_fit
from autolens.model.galaxy import galaxy as g
from autolens.lens import lens_data as ld
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import ray_tracing_plotters
from autolens.lens.plotters import lens_fit_plotters

# In this example, we'll fit the ccd data we simulated in the previous exercise. We'll do this using model
# images generated via a tracer, and by comparing to the simulated image we'll get diagostics about the quality of the fit.

# First you need to change the path below to the chapter 1 directory so we can load the data we output previously.
chapter_path = (
    "/home/jammy/PycharmProjects/PyAutoLens/workspace/howtolens/chapter_1_introduction/"
)

# The data path specifies where the data was output in the last tutorial, this time in the directory 'chapter_path/data'
data_path = chapter_path + "data/"

ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + "image.fits",
    noise_map_path=data_path + "noise_map.fits",
    psf_path=data_path + "psf.fits",
    pixel_scale=0.1,
)

# 'ccd_data' is a CCDData object, which is a 'package' of all components of the CCD instrument of the lens,
# in particular:

# 1) The image.
# 2) The Point Spread Function (PSF).
# 3) Its noise-map.

print("Image:")
print(ccd_data.image)
print("Noise-Map:")
print(ccd_data.noise_map)
print("PSF:")
print(ccd_data.psf)

# To fit an image, we first specify a mask. A mask describes the sections of the image that we fit.

# Typically, we want to mask regions of the image where the lens and source galaxies are not visible, for example
# at the edges where the signal is entirely background sky and noise.

# For the image we simulated, a 3" circular mask will do the job.

mask = ma.Mask.circular(
    shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0
)

print(mask)  # 1 = True, which means the pixel is masked. Edge pixels are indeed masked.
print(mask[48:53, 48:53])  # Whereas central pixels are False and therefore unmasked.

# We can use a ccd_plotter to compare the mask and the image - this is useful if we really want to 'tailor' a
# mask to the lensed source's light (which in this example, we won't).
ccd_plotters.plot_image(ccd_data=ccd_data, mask=mask)

# We can also use the mask to 'zoom' our plot around the masked region only - meaning that if our image is very large,
# we can focus-in on the lens and source galaxies.

# You'll see this is an option for pretty much every plotter in PyAutoLens, and is something we'll do often throughout
# the tutorials.
ccd_plotters.plot_image(ccd_data=ccd_data, mask=mask, zoom_around_mask=True)

# We can also remove all pixels outside the mask in the plot, meaning bright pixels outside the mask won't impact the
# plot's color range. Again, we'll do this throughout the code.
ccd_plotters.plot_image(
    ccd_data=ccd_data, mask=mask, extract_array_from_mask=True, zoom_around_mask=True
)

# Now we've loaded the ccd data and created a mask, we'll create a 'lens data' object, using the 'lens_data' module.

# A lens data object is a 'package' of all parts of a data-set we need in order to fit it with a lens model:

# 1) The ccd-data, including the image, PSF (so that when we compare a tracer's image to the image instrument we
#    can include blurring due to the telescope optics) and noise-map (so our goodness-of-fit measure accounts for
#    noise in the observations).

# 2) The mask, so that only the regions of the image with a signal are fitted.

# 3) A grid aligned to the ccd-imaging data's pixels, so the tracer's image is generated on the same (masked) grid as
# the image.

lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)

ccd_plotters.plot_image(ccd_data=lens_data.ccd_data)

# By printing its attribute, we can see that it does indeed contain the unmasked image, unmasked noise-map mask,
# psf and so on.
print("Image:")
print(lens_data.unmasked_image)
print()
print("Noise-Map:")
print(lens_data.unmasked_noise_map)
print()
print("PSF:")
print(lens_data.psf)
print()
print("Mask")
print(lens_data.mask_2d)
print()

# The lens_data also contains a masked image, returned below in 2D and 1D.

# On the 2D array, all edge values are masked and are therefore zeros. To see the image values, try changing the
# indexes of the array that are print to the central pixels (e.g. [49, 50])

print("The 2D Masked Image and 1D Image of unmasked entries")
print(lens_data.image(return_in_2d=True).shape)
print(lens_data.image(return_in_2d=True))
print(lens_data.image(return_in_2d=False).shape)
print(lens_data.image(return_in_2d=False))

# There is a masked noise-map, again returned in 2D and 1D with edge values set to zeros.

print("The 2D Masked Noise-Map and 1D Noise-Map of unmasked entries")
print(lens_data.noise_map(return_in_2d=True).shape)
print(lens_data.noise_map(return_in_2d=True))
print(lens_data.noise_map(return_in_2d=False).shape)
print(lens_data.noise_map(return_in_2d=False))

# The lens-data also has a grid, where only coordinates which are not masked are included.
print("Masked Grid")
print(lens_data.grid)

# To fit an image, create an image using a tracer. Lets use the same tracer we simulated the ccd instrument with (thus,
# our fit is 'perfect').

# Its worth noting that below, we use the lens_data's grid to setup the tracer. This ensures that our image-plane
# image is the same resolution and alignment as our lens data's masked image.

lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)

source_galaxy = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = ray_tracing.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

ray_tracing_plotters.plot_profile_image(tracer=tracer, grid=lens_data.grid)

# To fit the image, we pass the lens data and tracer to the 'lens_fit' module. This performs the following:

# 1) Blurs the tracer's image with the lens data's PSF, ensuring the telescope optics are included in the fit. This
#    creates the fit's 'model_image'.

# 2) Computes the difference between this model_image and the observed image-data, creating the fit's 'residual_map'.

# 3) Divides the residual-map by the noise-map and squares each value, creating the fit's 'chi_squared_map'.

# 4) Sums up these chi-squared values and converts them to a 'likelihood', which quantifies how good the tracer's fit
#    to the data was (higher likelihood = better fit).

fit = lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

lens_fit_plotters.plot_fit_subplot(
    fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True
)

# We can print the fit's attributes. As usual, we can choose whether to return the fits in 2d or 1d, and in 2d if we
# don't specify where we'll get all zeros, as the edges were masked:
print("Model-Image:")
print(fit.model_image(return_in_2d=True))
print(fit.model_image(return_in_2d=False))
print()
print("Residual Maps:")
print(fit.residual_map(return_in_2d=True))
print(fit.residual_map(return_in_2d=False))
print()
print("Chi-Squareds Maps:")
print(fit.chi_squared_map(return_in_2d=True))
print(fit.chi_squared_map(return_in_2d=False))

# Of course, the central unmasked pixels have non-zero values.
print("Model-Image Central Pixels:")
model_image = fit.model_image(return_in_2d=True)
print(model_image[48:53, 48:53])
print()

residual_map = fit.residual_map(return_in_2d=True)
print("Residuals Central Pixels:")
print(residual_map[48:53, 48:53])
print()

print("Chi-Squareds Central Pixels:")
chi_squared_map = fit.chi_squared_map(return_in_2d=True)
print(chi_squared_map[48:53, 48:53])

# The fit also gives a likelihood, which is a single-figure estimate of how good the model image fitted the
# simulated image (in unmasked pixels only!).
print("Likelihood:")
print(fit.likelihood)

# We used the same tracer to create and fit the image, giving an excellent fit. The residual-map and chi-squared-map,
# show no signs of the source galaxy's light present, indicating a good fit. This solution will translate to one of
# the highest-likelihood solutions possible.

# Lets change the tracer, so that it's near the correct solution, but slightly off. Below, we slightly offset the
# lens galaxy, by 0.005"

lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(
        centre=(0.005, 0.005), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)

source_galaxy = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = ray_tracing.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

lens_fit_plotters.plot_fit_subplot(
    fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True
)

# Residuals now appear at the locations of the source galaxy, increasing the chi-squared values (which determine our
# likelihood).

# Lets compare the likelihood to the value we computed above (which was 11697.24):
print("Previous Likelihood:")
print(11697.24)
print("New Likelihood:")
print(fit.likelihood)

# It decreases! As expected, this model us a worse fit to the data.

# Lets change the tracer, one more time, to a solution nowhere near the correct one.
lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(
        centre=(0.005, 0.005), einstein_radius=1.3, axis_ratio=0.8, phi=45.0
    ),
)

source_galaxy = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.7,
        phi=65.0,
        intensity=1.0,
        effective_radius=0.4,
        sersic_index=3.5,
    ),
)

tracer = ray_tracing.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

lens_fit_plotters.plot_fit_subplot(
    fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True
)

# Clearly, the model provides a terrible fit and this tracer is not a plausible representation of
# the image-data (of course, we already knew that, given that we simulated it!)

# The likelihood drops dramatically, as expected.
print("Previous Likelihoods:")
print(11697.24)
print(10319.44)
print("New Likelihood:")
print(fit.likelihood)

# Congratulations, you've fitted your first strong lens with PyAutoLens! Perform the following exercises:

# 1) In this example, we 'knew' the correct solution, because we simulated the lens ourselves. In the real Universe,
#    we have no idea what the correct solution is. How would you go about finding the correct solution?
#    Could you find a solution that fits the data reasonable through trial and error?
