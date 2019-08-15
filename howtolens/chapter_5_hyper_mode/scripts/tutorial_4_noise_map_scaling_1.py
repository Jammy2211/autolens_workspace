from autolens.data.instrument import abstract_data
from autolens.data.instrument import ccd
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing
from autolens.lens import lens_fit
from autolens.lens import lens_data as ld
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.lens.plotters import lens_fit_plotters
from autolens.model.inversion.plotters import inversion_plotters
from autolens.plotters import array_plotters

# So, in tutorial 1, we discussed how when our inversion didn't fit a compact source well we had skewed and undesirable
# chi-squared distribution. A small subset of the lensed source's brightest pixels were fitted poorly, contributing to
# the majority of our chi-squared signal. In terms of lens modeling, this meant that we would over-fit these regions of
# the image, which isn't ideal, as we would prefer that our lens model provides a global fit to the entire lensed
# source galaxy.

# With our adaptive pixelization and regularization, we were able to fit the instrument to the noise-limit and remove this
# skewed chi-squared distribution, so why do we need to introduce noise-map scaling? Well, we achieve a good fit when
# our lens's mass model is accurate. In the previous tutorials we used the *correct* lens mass model to demonstrate
# these features. But, what if our lens mass model isn't perfect? Well, we're gonna have residuals, and those residuals
# will cause the same problem as before; a skewed chi-squared distribution and an inability to fit the instrument to the
# noise level.

# So, lets simulate an image and fit it with a slightly incorrect mass model.

# This is the usual simulate function, using the compact source of the previous tutorials.


def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = abstract_data.PSF.from_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(
        shape=(150, 150), pixel_scale=0.05, sub_grid_size=2
    )

    lens_galaxy = g.Galaxy(
        redshift=0.5,
        mass=mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
        ),
    )

    source_galaxy = g.Galaxy(
        redshift=1.0,
        light=lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.7,
            phi=135.0,
            intensity=0.2,
            effective_radius=0.2,
            sersic_index=2.5,
        ),
    )

    tracer = ray_tracing.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack,
    )

    return ccd.SimulatedCCDData.from_tracer_and_exposure_arrays(
        tracer=tracer,
        pixel_scale=0.05,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=1.0,
        add_noise=True,
        noise_seed=1,
    )


# Lets simulate the instrument, draw a 3.0" mask and set up the lens instrument that we'll fit.

ccd_data = simulate()
mask = msk.Mask.circular(shape=(150, 150), pixel_scale=0.05, radius_arcsec=3.0)
lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)

# Next, we're going to fit the image using our magnification based grid. To perform the fits, we'll use a
# convenience function to fit the lens instrument we simulated above.

# In this fitting function, we have changed the lens galaxy's einstein radius to 1.55 from the 'true' simulated value
# of 1.6. Thus, we are going to fit the instrument with an *incorrect* mass model.


def fit_lens_data_with_source_galaxy(lens_data, source_galaxy):

    pixelization_grid = source_galaxy.pixelization.pixelization_grid_from_grid_stack(
        grid_stack=lens_data.grid_stack,
        hyper_image=source_galaxy.hyper_galaxy_image_1d,
        cluster=lens_data.cluster,
    )

    grid_stack_with_pixelization_grid = lens_data.grid_stack.new_grid_stack_with_grids_added(
        pixelization=pixelization_grid
    )

    lens_galaxy = g.Galaxy(
        redshift=0.5,
        mass=mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.55
        ),
    )

    tracer = ray_tracing.Tracer.from_galaxies(
        galaxies=[lens_galaxy, source_galaxy],
        image_plane_grid_stack=grid_stack_with_pixelization_grid,
        border=lens_data.border,
    )

    return lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)


# And now, we'll use the same magnification based source to fit this instrument.

source_magnification = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiMagnification(shape=(30, 30)),
    regularization=reg.Constant(coefficient=3.3),
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_magnification
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# Okay, so the fit isn't great. The main structure of the lensed source is reconstructed, but there are significant
# residuals left over. These residuals are a lot worse than we saw in the previous tutorials (when source's compact
# central structure was the problem). So, the obvious question is can our adaptive pixelization and regularization
# schemes address the problem?

# Lets find out, using this solution as our hyper_galaxy-image. In this case, our hyper_galaxy-image isn't a perfect fit to the instrument.
# This isn't ideal, but shouldn't be to problematic either. Although the solution leaves residuals it still
# captures the source's overall structure. The pixelization / regularization hyper_galaxy-parameters have enough flexibility
# in how they use this image to adapt themselves, so the hyper_galaxy-image doesn't *need* to be perfect.

hyper_image_1d = fit.model_image(return_in_2d=False)

# You'll note that, unlike before, this source galaxy receives two types of hyper_galaxy-images, a 'hyper_galaxy_image'
# (like before) and a 'hyper_model_image' (which is new). I'll come back to this later.

source_adaptive = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image_1d=hyper_image_1d,
    hyper_model_image_1d=hyper_image_1d,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_adaptive
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

print("Evidence = ", fit.evidence)

# Okay, so the solution is better, but its far from perfect. Furthermore, this solution maximizes the Bayesian evidence.
# This means there is no reasonable way to change our source pixelization or regularization to better fit the instrument.
# There is a fundamental problem with our lens's mass model!

# To remind you of what we discussed in tutorial 1, this poses a major problem for our model-fitting. Because a small
# subset of our instrument has such large chi-squared values, the non-linear search is going to seek solutions which reduce
# only these chi-squared values. For the image above, this means that a small subset of our instrument (e.g. < 5% of pixels)
# contributes to the majority of our likelihood (e.g. > 95% of the overall chi-squared). This is *not* what we want,
# as it means that instead of using the entire surface brightness profile of the lensed source galaxy to constrain our
# lens model, we end up using only a small subset of its brightest pixels.

# This is even more problematic when we try and use the Bayesian evidence to objectively quantify the quality of the
# fit, as it means it cannot obtain a solution that provides a reduced chi-squared of 1.

# So, you're probably wondering, whats the problem? Can't we just change the mass model to fit the instrument better? Surely
# if we actually modeled this lens with PyAutoLens, it wouldn't go to this solution anyway, but instead infer the
# correct Einstein radius of 1.6?

# That's true. However, for *real* strong gravitational lenses, there is no such thing as a 'correct mass model'.
# Real galaxies are not EllipticalIsothermal profiles, or power-laws, or NFW's, or any of the symmetric and smooth
# analytic profiles we assume to model their mass. This means that for real strong lens imaging we could *always*
# end up in the situation where our source-reconstruction leaves residuals, producing these skewed chi-squared
# distributions, and that PyAutoLens can't remove them by simply improving the mass model.

# This is where noise-map scaling comes in. If we have no alternative, the best way to get Gaussian-distribution (e.g.
# more uniform) chi-squared fit is to increase the variances of image pixels with high chi-squared values. So, that's
# what we're going to do, by making our source galaxy a 'hyper_galaxy-galaxy', that is, a galaxy which use's its hyper_galaxy
# image's to increase the noise in pixels where it has a large signal. Let take a look.

source_hyper_galaxy = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy=g.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.5, noise_power=1.0
    ),
    hyper_galaxy_image_1d=hyper_image_1d,
    hyper_model_image_1d=hyper_image_1d,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_hyper_galaxy
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# As expected, the chi-squared distribution looks *alot* better. The chi-squareds have reduced from the 200's to the
# 50's, because the variances were increased. This is what we want, so lets make sure we see an appropriate increase in
# Bayesian evidence

print("Evidence using baseline variances = ", 8911.66)

print("Evidence using variances hyper by hyper_galaxy galaxy = ", fit.evidence)

# Yep, a huge increase in the 1000's! Clearly, if our model doesn't fit the instrument well, we *need* to increase the noise
# wherever the fit is poor to ensure that our use of the Bayesian evidence is well defined.

# So, how does the HyperGalaxy that we attached to the source-galaxy above actually scale the noise?

# First, it creates a 'contribution_map' from the hyper_galaxy-image of the lensed source galaxy. This also uses the
# 'hyper_model_image', which is the overall model-image of the best-fit lens model. In this tutorial, because our
# strong lens imaging only has a source galaxy emitting light, the hyper_galaxy-image of the source galaxy is the same as the
# hyper_model_image. However, In the next tutorial, we'll introduce the lens galaxy's light, such that each hyper_galaxy
# galaxy image is different to the hyper_galaxy model image!

# We compute the contribution map as follows:

# 1) Add the 'contribution_factor' hyper_galaxy-parameter value to the 'hyper_model_image'.

# 2) Divide the 'hyper_galaxy_image' by the hyper_galaxy-model image created in step 1).

# 3) Divide the image created in step 2) by its maximum value, such that all pixels range between 0.0 and 1.0.

# Lets look at a few contribution maps, generated using hyper_galaxy-galaxy's with different contribution factors.

source_contribution_factor_1 = g.Galaxy(
    redshift=1.0,
    hyper_galaxy=g.HyperGalaxy(contribution_factor=1.0),
    hyper_galaxy_image_1d=hyper_image_1d,
    hyper_model_image_1d=hyper_image_1d,
)

contribution_map_1d = source_contribution_factor_1.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image_1d, hyper_galaxy_image=hyper_image_1d
)

contribution_map_2d = mask.scaled_array_2d_from_array_1d(array_1d=contribution_map_1d)

array_plotters.plot_array(
    array=contribution_map_2d,
    title="Contribution Map",
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

source_contribution_factor_3 = g.Galaxy(
    redshift=1.0,
    hyper_galaxy=g.HyperGalaxy(contribution_factor=3.0),
    hyper_galaxy_image_1d=hyper_image_1d,
    hyper_model_image_1d=hyper_image_1d,
)

contribution_map_1d = source_contribution_factor_3.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image_1d, hyper_galaxy_image=hyper_image_1d
)

contribution_map_2d = mask.scaled_array_2d_from_array_1d(array_1d=contribution_map_1d)

array_plotters.plot_array(
    array=contribution_map_2d,
    title="Contribution Map",
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

source_hyper_galaxy = g.Galaxy(
    redshift=1.0,
    hyper_galaxy=g.HyperGalaxy(contribution_factor=5.0),
    hyper_galaxy_image_1d=hyper_image_1d,
    hyper_model_image_1d=hyper_image_1d,
)

contribution_map_1d = source_hyper_galaxy.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image_1d, hyper_galaxy_image=hyper_image_1d
)

contribution_map_2d = mask.scaled_array_2d_from_array_1d(array_1d=contribution_map_1d)

array_plotters.plot_array(
    array=contribution_map_2d,
    title="Contribution Map",
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# So, by increasing the contribution factor, we allocate more pixels with higher contributions (e.g. values closer to
# 1.0) than pixels with lower values. Practically, this is all the contribution_factor does; it gives a means by which
# to scale how we allocate contributions to the source galaxy. Now, we're going to use this contribution map to
# scale the noise-map, as follows:

# 1) Multiply the baseline (e.g. unscaled) noise-map of the image-instrument by the contribution map made in step 3) above.
#    This means that only noise-map values where the contribution map has large values (e.g. near 1.0) are going to
#    remain in this image, with the majority of values multiplied by contribution map values near 0.0.

# 2) Raise the noise-map generated in step 1) above to the power of the hyper_galaxy-parameter noise_power. Thus, for large
#    values of noise_power, the largest noise-map values will be increased even more, raising their noise the most.

# 3) Multiply the noise-map values generated in step 2) by the hyper_galaxy-parameter noise_factor. Again, this is a means by
#    which PyAutoLens is able to scale the noise-map values.

# Lets compare two fits, one where a hyper_galaxy-galaxy scales the noise-map, and one where it dooesn't.

source_no_hyper_galaxy = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image_1d=hyper_image_1d,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_no_hyper_galaxy
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

print("Evidence using baseline variances = ", fit.evidence)

source_hyper_galaxy = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy=g.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.5, noise_power=1.0
    ),
    hyper_galaxy_image_1d=hyper_image_1d,
    hyper_model_image_1d=hyper_image_1d,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_hyper_galaxy
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

print("Evidence using variances hyper by hyper_galaxy galaxy = ", fit.evidence)

# Feel free to play around with the noise_factor and noise_power hyper_galaxy-parameters above. It should be fairly clear
# what they do; they simply change the amount by which the noise is increased.

# And with that, we've completed the first of two tutorials on noise-map scaling. To end, I want you to have a quick
# think, is there anything else that you can think of that would mean we need to scale the noise? In this tutorial, it
# was the inadequacy of our mass-model that lead to significant residuals and a skewed chi-squared distribution.
# What else might cause residuals? I'll give you a couple below;

# 1) A mismatch between our model of the imaging data's Point Spread Function (PSF) and the true PSF of the telescope
#    optics of the instrument.

# 2) Unaccounted for effects in our instrument-reduction for the image, in particular the presense of correlated signal and
#    noise during the image's instrument reduction.

# 3) A sub-optimal background sky subtraction of the image, which can leave large levels of signal in the outskirts of
#    the image that are not due to the strong lens system itself.

# Oh, there's on more thing that can cause much worse residuals than all the effects above. That'll be the topic of
# the next tutorial.
