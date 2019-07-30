from autolens.data import ccd
from autolens.data import simulated_ccd
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
from autolens.plotters import array_plotters

# Okay, so noise-map scaling is important when our mass model means our source reconstruction is inaccurate. However,
# it serves an even more important use, when some other component of our lens model doesn't fit the data well.
# Can you think what it is? What could leave significant residuals in our model-fit? And what might happen to also be
# the highest S/N values in our image, meaning these residuals contribute *even more* to the chi-squared distribution?

# Yep, you guessed it, it's the lens galaxy light profile fit and subtraction. Just like our overly simplified
# mass profile's mean we can't perfectly reconstruct the source's light, the same holds true of the Sersic profiles
# we use to fit the lens galaxy's light. Lets take a look.

# This simulates the exact same data as the previous tutorial, but with the lens light included.


def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(
        shape=(150, 150), pixel_scale=0.05, sub_grid_size=2
    )

    lens_galaxy = g.Galaxy(
        redshift=0.5,
        light=lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            phi=45.0,
            intensity=0.5,
            effective_radius=0.8,
            sersic_index=3.0,
        ),
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

    tracer = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
        galaxies=[lens_galaxy, source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack,
    )

    return simulated_ccd.SimulatedCCDData.from_tracer_and_exposure_arrays(
        tracer=tracer,
        pixel_scale=0.05,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=1.0,
        add_noise=True,
        noise_seed=1,
    )


# Lets simulate the data with lens light, draw a 3.0" mask and set up the lens data that we'll fit.

ccd_data = simulate()
mask = msk.Mask.circular(shape=(150, 150), pixel_scale=0.05, radius_arcsec=3.0)
lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)

# Again, we'll use a convenience function to fit the lens data we simulated above.


def fit_lens_data_with_lens_and_source_galaxy(lens_data, lens_galaxy, source_galaxy):

    pixelization_grid = source_galaxy.pixelization.pixelization_grid_from_grid_stack(
        grid_stack=lens_data.grid_stack,
        hyper_image=source_galaxy.hyper_galaxy_image_1d,
        cluster=lens_data.cluster,
    )

    grid_stack_with_pixelization_grid = lens_data.grid_stack.new_grid_stack_with_grids_added(
        pixelization=pixelization_grid
    )

    tracer = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
        galaxies=[lens_galaxy, source_galaxy],
        image_plane_grid_stack=grid_stack_with_pixelization_grid,
        border=lens_data.border,
    )

    return lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)


# Now, lets use this function to fit the lens data. We'll use a lens model with the correct mass model, but an
# incorrect lens light profile. The source will use a magnification based grid.

lens_galaxy = g.Galaxy(
    redshift=0.5,
    light=lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=0.4,
        effective_radius=0.8,
        sersic_index=3.0,
    ),
    mass=mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
    ),
)


source_magnification = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiMagnification(shape=(30, 30)),
    regularization=reg.Constant(coefficient=3.3),
)

fit = fit_lens_data_with_lens_and_source_galaxy(
    lens_data=lens_data, lens_galaxy=lens_galaxy, source_galaxy=source_magnification
)

print("Evidence using baseline variances = ", fit.evidence)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# Okay, so its clear that our poor lens light subtraction leaves clear residuals in the centre of the lens galaxy.
# Because these pixels are extremely high S/N, they contribute very large values to the chi-squared. Whatsmore, for a
# real strong lens, we could not fit these residual features using a more complex light profile. These types of
# residuals are extremely common, and they are caused by nasty, irregular morphological structures in the lens galaxy;
# nuclear star emission, nuclear rings, bars, bulges, and what not.

# So, this skewed chi-squared distribution will cause all the same problems we discussed in the previous tutorial, like
# over-fitting. However, in terms of the source-reconstruction and Bayesian evidence, the residuals are way more
# problematic then the previous chapter. Why? Because when we compute the Bayesian evidence for the source-inversion,
# these pixels are included like all the other image pixels. But, *they do not contain the source*. The Bayesian
# evidence is going to try improve the fit to these pixels by reducing the level of regularization,  but its *going to
# fail miserably*, as they map nowhere near the source!

# This is a fundamental problem when simultaneously modeling the lens galaxy's light and source galaxy using an
# inversion. The inversion has no way to distinguish whether the flux it is reconstructing belongs to the lens or
# source. This is why contribution maps, introduced in the previous tutorial, are so valuable; by creating a
# contribution map for every galaxy in the image, PyAutoLens has a means by which to distinguish which flux in what
# pixels belongs to each component in the image! This is further aided by the pixelizations / regularizations
# that adapt to the source morphology, as not only are they adapting to where the source *is*, they adapt to where
# *it isn't* (and therefore where the lens galaxy is), by changing the source-pixel sizes and regularization.

# Okay, so now, lets create our hyper-images and use them create the contribution maps of our lens and source galaxies.
# Note below that we now create separate model images for our lens and source galaxies. This is what will allow us to
# create contribution maps for each.

hyper_image_1d = fit.model_image(return_in_2d=False)

hyper_image_lens_2d = fit.model_image_2d_of_planes[
    0
]  # This is the model image of the lens
hyper_image_lens_1d = mask.array_1d_from_array_2d(array_2d=hyper_image_lens_2d)

hyper_image_source_2d = fit.model_image_2d_of_planes[
    1
]  # This is the model image of the source
hyper_image_source_1d = mask.array_1d_from_array_2d(array_2d=hyper_image_source_2d)

lens_galaxy_hyper = g.Galaxy(
    redshift=0.5,
    light=lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=0.4,
        effective_radius=0.8,
        sersic_index=3.0,
    ),
    mass=mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
    ),
    hyper_galaxy=g.HyperGalaxy(
        contribution_factor=0.3, noise_factor=4.0, noise_power=1.5
    ),
    hyper_model_image_1d=hyper_image_1d,
    hyper_galaxy_image_1d=hyper_image_lens_1d,  # <- The lens get its own hyper image.
)

source_magnification_hyper = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiMagnification(shape=(30, 30)),
    regularization=reg.Constant(coefficient=3.3),
    hyper_galaxy=g.HyperGalaxy(
        contribution_factor=2.0, noise_factor=2.0, noise_power=3.0
    ),
    hyper_galaxy_image_1d=hyper_image_1d,
    hyper_model_image_1d=hyper_image_source_1d,  # <- The source get its own hyper image.
)

fit = fit_lens_data_with_lens_and_source_galaxy(
    lens_data=lens_data, lens_galaxy=lens_galaxy, source_galaxy=source_magnification
)

lens_contribution_map_1d = lens_galaxy_hyper.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image_1d, hyper_galaxy_image=hyper_image_lens_1d
)

lens_contribution_map_2d = mask.scaled_array_2d_from_array_1d(
    array_1d=lens_contribution_map_1d
)

array_plotters.plot_array(
    array=lens_contribution_map_2d,
    title="Lens Contribution Map",
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

source_contribution_map_1d = source_magnification_hyper.hyper_galaxy.contribution_map_from_hyper_images(
    hyper_model_image=hyper_image_1d, hyper_galaxy_image=hyper_image_source_1d
)

source_contribution_map_2d = mask.scaled_array_2d_from_array_1d(
    array_1d=source_contribution_map_1d
)

array_plotters.plot_array(
    array=source_contribution_map_2d,
    title="Source Contribution Map",
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# Okay, so clearly the contribution maps successfully decompose the image into its different components. Now, we can
# use each contribution map to scale different regions of the noise-map. This is key, as from the fit above, it was
# clear that both the lens and source required the noise to be scaled, but they had different chi-squared values
# ( > 150 and ~ 30), meaning they required different levels of noise-scaling. Lets see how much our fit improves
# and Bayesian evidence increases when we include noise-scaling

fit = fit_lens_data_with_lens_and_source_galaxy(
    lens_data=lens_data,
    lens_galaxy=lens_galaxy_hyper,
    source_galaxy=source_magnification_hyper,
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

print("Evidence using baseline variances = ", 8861.51)

print("Evidence using hyper-galaxy scaled variances = ", fit.evidence)

# Great, and with that, we've covered hyper galaxies. You might be wondering, what happens if there are multiple lens
# galaxies? or multiple source galaxies? Well, as you'd expect, PyAutoLens will make each a hyper-galaxy, and therefore
# scale the noise-map of that individual galaxy in the image. This is what we want, as different parts of the image
# require different levels of noise-map scaling.

# Finally, I want to quickly mention two more ways that we change our data during th fitting process, one which scales
# the background noise and one which scales the background sky in the image. To do this, we use the 'hyper_data'
# module in PyAutoLens.

from autolens.model.hyper import hyper_data as hd

# This module includes all components of the model that scale parts of the data. To scale the background sky in the
# image we use the HyperImageSky class, and input a 'background_sky_scale'.

hyper_image_sky = hd.HyperImageSky(background_sky_scale=1.0)

# The background_sky_scale is literally just a constant value we add to every pixel of the observed image before
# fitting it, therefore increasing or decreasing the background sky level in the image .This means we can account for
# an inaccurate background sky subtraction in our data reduction during the PyAutoLens model fitting.

# We can also scale the background noise in an analogous fashion, using the HyperNoiseBackground class and the
# 'background_noise_scale' hyper-parameter. This value is added to every pixel in the noise-map.

hyper_noise_background = hd.HyperNoiseBackground(background_noise_scale=1.0)

# To use these hyper-data parameters, we pass them to a lens-fit just like we do our tracer.

pixelization_grid = source_magnification_hyper.pixelization.pixelization_grid_from_grid_stack(
    grid_stack=lens_data.grid_stack,
    hyper_image=source_magnification_hyper.hyper_galaxy_image_1d,
    cluster=lens_data.cluster,
)

grid_stack_with_pixelization_grid = lens_data.grid_stack.new_grid_stack_with_grids_added(
    pixelization=pixelization_grid
)

tracer = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
    galaxies=[lens_galaxy_hyper, source_magnification_hyper],
    image_plane_grid_stack=grid_stack_with_pixelization_grid,
    border=lens_data.border,
)

lens_fit.LensDataFit.for_data_and_tracer(
    lens_data=lens_data,
    tracer=tracer,
    hyper_image_sky=hyper_image_sky,
    hyper_noise_background=hyper_noise_background,
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# Is there any reason to scale the background noise, other than if the background sky subtraction has a large
# correction? There is. Basically, there are a lot of pixels in our image which do not contain the lensed source, but
# are fitted by the inversion. As we've learnt in this chapter, this isn't problematic when we have our adaptive
# regularization scheme because the regularization coefficient will be increased to large values.

# However, if you ran a full PyAutoLens analysis in hyper-mode (which we cover in the next tutorial), you'd find the
# method still dedicates a lot of source-pixels to fit these regions of the image, _even though they have no source_.
# Why is this? Well, its because although these pixels have no source, they still have a relatively high S/N value
# (of order 5-10) due to the lens galaxy (e.g. its flux before it is subtracted). The inversion when reconstructing
# the data 'sees' pixels with a S/N > 1 and achieves a higher Bayesian evidence by fitting these pixel's flux.

# But, if we increase the background noise, then these pixels will go to much lower S/N values (<  1). Then, the
# adaptive pixelization will feel no need to fit them properly, and begin to fit these regions of the source-plane
# with far fewer, much bigger source pixels! This will again give us a net increase in Bayesian evidence, but more
# importantly, it will dramatically reduce the number of source pixels we use to fit the data. And what does fewer
# soruce-pixels mean? Much, much faster run times. Yay!

# With that, we have introduced every feature of hyper-mode. The only thing left for us to do is to bring it all
# together, and consider how we use all of these features in PyAutoLens pipelines. That is what we'll discuss in the
# next tutorial, and then you'll be ready to perform your own hyper-fits!
