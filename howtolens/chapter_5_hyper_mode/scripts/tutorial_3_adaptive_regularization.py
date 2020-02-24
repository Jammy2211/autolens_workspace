import autolens as al
import autolens.plot as aplt

# In tutorial 1, we considered why our Constant regularization scheme was sub-optimal. Diffferent regions of the source
# demand different levels of regularization, motivating a regularization scheme which adapts to the reconstructed
# source's surface brightness.

# This raises the same question as before, how do we adapt our regularization scheme to the source before we've
# recontructed it? Just like in the last tutorial, we'll use a model image of a strongly lensed source from a previous
# phase of the pipeline that we've begun calling the 'hyper-galaxy-image'.

# This is the usual simulate function, using the compact source of the previous tutorials.


def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.7,
            phi=135.0,
            intensity=0.2,
            effective_radius=0.2,
            sersic_index=2.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(150, 150),
        pixel_scales=0.05,
        exposure_time=300.0,
        sub_size=2,
        psf=psf,
        background_level=1.0,
        add_noise=True,
        noise_seed=1,
    )

    return simulator.from_tracer(tracer=tracer)


# Lets simulate the dataset, draw a 3.0" mask and set up the lens dataset that we'll fit.

imaging = simulate()
mask = al.mask.circular(shape_2d=(150, 150), pixel_scales=0.05, radius=3.0)
masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

# Next, we're going to fit the image using our magnification based grid. To perform the fits, we'll use a convenience
# function to fit the lens dataset we simulated above.


def fit_masked_imaging_with_source_galaxy(masked_imaging, source_galaxy):

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.fit(masked_dataset=masked_imaging, tracer=tracer)


# Next, we'll use the magnification based source to fit this simulator.

source_magnification = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=3.3),
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_magnification
)

aplt.fit_imaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

aplt.inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)


# Okay, so the inversion's fit looks just like it did in the previous tutorials. Lets quickly remind ourselves that the
# effective regularization coefficient of each source pixel is our input coefficient value of 3.3.

aplt.inversion.regularization_weights(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

# Lets now look at adaptive regularization in action, by setting up a hyper-galaxy-image and using the
# 'AdaptiveBrightness' regularization scheme. This introduces additional hyper-galaxy-parameters, that I'll explain next.

hyper_image = fit.model_image.in_1d

source_adaptive_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.005, outer_coefficient=1.9, signal_scale=3.0
    ),
    hyper_galaxy_image=hyper_image,
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_adaptive_regularization
)

aplt.inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

aplt.inversion.regularization_weights(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

# So, as expected, we now have a variable regularization scheme. The regularization of the source's brightest regions
# is much lower than that of its outer regions. As discussed before, this is what we want. Lets quickly check that this
# does, indeed, increase the Bayesian evidence:

print("Evidence using constant regularization = ", 14236.292117135737)

print("Evidence using adaptive regularization = ", fit.evidence)

# Yep, and we again get an increase beyond 200! Of course, combining the adaptive pixelization and regularization
# will only further benefit our lens modeling!

# However, as shown below, we don't fit the source as well as the morphology based pixelization did in the last
# chapter. This is because although the adaptive regularization scheme improves the fit, the magnification based grid
# simply *does not*  have sufficient resolution to resolve the source's cuspy central light profile.

aplt.fit_imaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

# So, how does adaptive regularization work?

# For every source-pixel, we have a mapping between that pixel and a set of pixels in the hyper-galaxy-image. Therefore,
# for every source-pixel, if we sum the values of all hyper-galaxy-image pixels that map to it we get an estimate of how
# much of the lensed source's signal we expect will be reconstructed. We call this each pixel's 'pixel signal'.

# If a source-pixel has a higher pixel-signal, we anticipate that it'll reconstruct more flux and we use this
# information to regularize it less. Conversely, if the pixel-signal is close to zero, the source pixel will reconstruct
# near-zero flux and regularization will smooth over these pixels by using a high regularization coefficient.

# This works as follows:

# 1) For every source pixel, compute its pixel-signal, the summed flux of all corresponding image-pixels in the
#    hyper-galaxy-image.

# 2) Divide every pixel-signal by the number of image-pixels that map directly to that source-pixel. In doing so, all
#    pixel-signals are 'relative'. This means that source-pixels which by chance map to more image-pixels than their
#    neighbors will not have a higher pixel-signal, and visa versa. This ensures the specific pixelization grid does
#    impact the adaptive regularization pattern.

# 3) Divide the pixel-signals by the maximum pixel signal so that they range between 0.0 and 1.0.

# 4) Raise these values to the power of the hyper-galaxy-parameter *signal_scale*. For a *signal_scale* of 0.0, all
#    pixels will therefore have the same final pixel-scale. As the *signal_scale* increases, a sharper transition of
#    of pixel-signal values arises between regions with high and low pixel-signals.

# 5) Compute every source pixel's effective regularization coefficient as:

#    (inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)) ** 2.0

#    This uses two regularization coefficients, one which is applied to pixels with high pixel-signals and one
#    to pixels with low pixel-signals. Thus, pixels in the inner regions of the source may be given
#    a lower level of regularization than pixels further away, as desired.

# Thus, we now adapt our regularization scheme to the source's surface brightness. Where its brighter (and therefore
# has a steeper flux gradient) we apply a lower level of regularization than further out. Furthermore, in the edges of
# the source-plane where no source-flux is present we will assume a high regularization coefficients that smooth over
# all source-pixels.

# Try looking at a couple of extra solutions which use with different inner and outer regularization coefficients or
# signal scales. I doubt you'll notice a lot change visually, but the evidence certainly has a lot of room for manoveur
# with different values.

# You may find solutions that raise an 'InversionException'. These solutions mean that the matrix used during the linear
# algebra calculation was ill-defined, and could not be inverted. These solutions are removed by PyAutoLens during lens modeling.

source_adaptive_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
    hyper_galaxy_image=hyper_image,
)

fit = fit_masked_imaging_with_source_galaxy(
    masked_imaging=masked_imaging, source_galaxy=source_adaptive_regularization
)

aplt.inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

aplt.inversion.regularization_weights(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

aplt.fit_imaging.subplot_fit_imaging(
    fit=fit, include=aplt.Include(inversion_image_pixelization_grid=True, mask=True)
)

print("Evidence using adaptive regularization = ", fit.evidence)

# To end, lets consider what this adaptive regularization scheme means in the context of maximizing the Bayesian
# evidence. In the previous tutorial, we noted that by using a brightness-based adaptive pixelization we
# increased the Bayesian evidence by allowing for new solutions which fit the dataset user fewer source pixels; the
# key criteria in making a source reconstruction 'more simple' and 'less complex'.

# As you might of guessed, adaptive regularization again increases the Bayesian evidence by making the source
# reconstruction simpler:

# 1) Reducing regularization in the source's brightest regions produces a 'simpler' solution in that we are not
#    over-smoothing our reconstruction of its brightest regions.

# 2) Increasing regularization in the outskirts produces a simpler solution by correlating more source-pixels,
#    effectively reducing the number of pixels used by the reconstruction.

# Together, brightness based pixelizations and regularization allow us to find the objectvely 'simplest' source solution
# possible and therefore ensure that our Bayesian evidence's have a well defined maximum value they are seeking. This
# was not the case for magnification based pixelizations and constant regularization schemes.
