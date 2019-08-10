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
from autolens.model.inversion.plotters import inversion_plotters
from autolens.lens.plotters import lens_fit_plotters

# In tutorial 1, we considered why our Constant regularization scheme was sub-optimal. Basically, different regions of
# the source demand different levels of regularization, motivating a regularization scheme which adapts to the
# reconstructed source's surface brightness.

# Just like the last tutorial, this raises a question, how do we adapt our regularization scheme to the source, before
# we've recontructed it? Just like in the last tutorial, we'll use a model image of a strongly lensed source from
# a previous phase of the pipeline, that we've begun calling the 'hyper-image'.

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

    tracer = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
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

# Next, we're going to fit the image using our magnification based grid. To perform the fits, we'll use another
# convenience function to fit the lens instrument we simulated above.


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
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
        ),
    )

    tracer = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
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


# Okay, so the inversion's fit looks just like it did in the previous tutorials. Lets quickly remind ourselves that the
# effective regularization coefficient of each source pixel is our input coefficient value of 3.3.

inversion_plotters.plot_pixelization_regularization_weights(
    inversion=fit.inversion, should_plot_centres=True
)

# Okay, so now lets look at adaptive regularization in action, by setting up a hyper-image and using the
# 'AdaptiveBrightness' regularization scheme. This introduces additional hyper-parameters, that I'll explain next.

hyper_image_1d = fit.model_image(return_in_2d=False)

source_adaptive_regularization = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiMagnification(shape=(30, 30)),
    regularization=reg.AdaptiveBrightness(
        inner_coefficient=0.005, outer_coefficient=1.9, signal_scale=3.0
    ),
    hyper_galaxy_image_1d=hyper_image_1d,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_adaptive_regularization
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

inversion_plotters.plot_pixelization_regularization_weights(
    inversion=fit.inversion, should_plot_centres=True
)

# So, as expected, we now have a variable regularization scheme. The regularization of the source's brightest regions
# is much lower than that of its outer regions. As discussed before, this is what we want. Lets quickly check that this
# does, indeed, increase the Bayesian evidence:

print("Evidence using constant regularization = ", 14236.292117135737)

print("Evidence using adaptive regularization = ", fit.evidence)

# Yep, and we again get an increase beyond 200, nice! Of course, combining the adaptive pixelization and regularization
# will only further benefit our lens modeling!

# However, as shown below, we don't fit the source as well as the morphology based pixelization did in the last
# chapter. This is because although the adaptive regularization scheme does help us better fit the instrument, it cannot
# change the fact we simply *do not* have sufficient resoluton to resolve its cuspy central light profile when using a
# magnification based grid.

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# Okay, so how does this adaptive regularization scheme work?

# For every source-pixel, we have a mapping between that pixel and a set of pixels in the hyper-image. Therefore,
# for every source-pixel, if we sum the values of all hyper-image pixels that map to it we get an estimate of how much
# of the lensed source's signal we expect will be reconstructed. We call this each pixel's 'pixel signal'.

# From here, the idea is simple, if source-pixels have a higher pixel-signal we use this information to regularize it
# less. Conversely, if the pixel-signal is close to zero, regularization smooths over these pixels by using a high
# regularization coefficient.

# This works as follows:

# 1) For every source pixel, compute its pixel-signal, that is the summed flux of all corresponding image-pixels in
#    the hyper-image.

# 2) Divide every pixel-signal by the number of image-pixels that map directly to that source-pixel. In doing so, all
#    pixel-signals are relative. This means that source-pixels which by chance map to more image-pixels than their
#    neighbors will not have a higher pixel-signal, and visa versa. This ensures the gridding of the pixelization does
#    not lead to a lack of smoothness in the adaptive regularization pattern.

# 3) Divide the pixel-signals by the maximum pixel signal, so that they range between 0.0 and 1.0.

# 4) Raise these values to the power of the hyper-parameter *signal_scale*. For a *signal_scale* of 0.0, all pixels
#    will therefore have the same final pixel-scale. As the *signal_scale* increases, a sharper transition of
#    of pixel-signal values arises between regions with high and low pixel-signals.

# 5) Compute every source pixel's effective regularization coefficient as:

#    (inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)) ** 2.0

#    This uses two regularization coefficients, one which is applied to pixels with high pixel-signals and one
#    to pixels with low pixel-signals. Thus, pixels in the inner regions of the source may be given
#    a lower level of regularization than pixels further away, as desired.

# Thus, we are able to now adapt our regularization scheme to the source's surface brightness. Where its brighter (and
# therefore where its flux has a steeper gradient) we can apply a much lower level of regularization than further out.
# Furthermore, in the extreme edges of the source-plane where no source-flux is present whatsoever, we can assume
# extremely high regularization coefficients that complete smooth over all source-pixels.

# Its worth noting that we also don't force the outer coefficient to be larger than the inner coefficient. So, if
# for some perverse strong lens the Evidence wanted the source's central light to be regularizated more than its
# outskirts, this solution is entirely possible! I'm yet to experience such a result in any lenses that I've
# modeled, but it isn't beyond the realms of possibility for sure!

# Feel free to look at a couple of extra solutions which use regularization schemes with different inner and outer
# coefficients or signal scales. I doubt you'll notice a lot change visually, but the evidence certainly has a
# lot of room for manoveur with different values.

# (You may also find solutions that raise an 'InversionException'. These solutions basically mean that the matrix
# used during the linear algebra calculation was ill-defined, and could not be inverted. These solultions are
# removed by PyAutoLens during lens modeling, so are non-consequetial).

source_adaptive_regularization = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiMagnification(shape=(30, 30)),
    regularization=reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image_1d=hyper_image_1d,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_adaptive_regularization
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

inversion_plotters.plot_pixelization_regularization_weights(
    inversion=fit.inversion, should_plot_centres=True
)

lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

print("Evidence using adaptive regularization = ", fit.evidence)

# To end, lets consider what this adaptive regularization scheme means in the context of maximizing the Bayesian
# evidence. In the previous tutorial, we noted that by using a brightness-based adaptive pixelization we
# increased the Bayesian evidence by allowing for new solutions which fit the instrument user fewer source pixels; the
# key criteria in making a source reconstruction 'more simple' and 'less complex'.

# As you might of guessed, adaptive regularization again increases the Bayesian evidence by making the source
# reconstruction simpler. What is actually getting simpler:

# 1) Reducing regularization in the source's brightest regions produces a 'simpler' solution in that we are not
#    over-smoothing our reconstruction of its brightest regions.

# 2) Increasing regularization in the outskirts produces a simpler solution by correlating more source-pixels,
#    effectively reducing the number of pixels used by the reconstruction.

# Together, brightness based pixelizations and regularization allow us to find the objectvely 'simplest' source solution
# possible, and therefore ensure that our Bayesian evidence's have a well defined maximum value they are seeking. This
# was not the case for magnification based pixelizations and constant regularization schemes.
