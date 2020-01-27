import autolens as al
import autolens.plot as aplt

# I think you'll agree, inversions are a very powerful tool for modeling strong lenses. Now that our source galaxies
# comprise just a few parameters we've got a much less complex non-linear parameter space to deal with. This allows us
# to fit more complex mass models and ask ever more interesting scientific questions!

# However, inversions arn't perfect, especially when we use to them model lenses. These arn't huge issues and they're
# easy to deal with, but its worth me explaining them now so they don't trip you up when you start using inversions!

# So, what happens if we fit an image using an inversion and the wrong lens model? lets simulate an image and find out.

# This is the usual simulator function.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.1, 0.1),
            axis_ratio=0.8,
            phi=90.0,
            intensity=0.2,
            effective_radius=0.3,
            sersic_index=1.0,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(180, 180),
        pixel_scales=0.05,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


# And the same fitting function as the last tutorial.
def perform_fit_with_lens__source_galaxy(lens_galaxy, source_galaxy):

    imaging = simulate()
    mask = al.mask.circular_annular(
        shape_2d=imaging.shape_2d,
        pixel_scales=imaging.pixel_scales,
        inner_radius=0.5,
        outer_radius=2.2,
    )

    masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.fit(masked_dataset=masked_imaging, tracer=tracer)


# This fit uses a lens galaxy with the wrong mass-model (I've reduced its Einstein Radius from 1.6 to 0.8).
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=0.8
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

fit = perform_fit_with_lens__source_galaxy(
    lens_galaxy=lens_galaxy, source_galaxy=source_galaxy
)

aplt.fit_imaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))

# What happened!? This incorrect mass-model provides a really good_fit to the image! The residuals and chi-squared map
# are as good as the ones we saw in the last tutorial.

# How can an incorrect lens model provide such a fit? Well, as I'm sure you noticed, the source has been reconstructed
# as a demagnified version of the image. Clearly, this isn't a physical solution or a solution that we want our
# non-linear search to find, but for inversions these solutions nevertheless exist.

# This isn't necessarily problematic for lens modeling. Afterall, the source reconstruction above is extremely complex,
# in that it requires a lot of pixels to fit the image accurately. Indeed, its Bayesian evidence is much lower than the
# correct solution.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

correct_fit = perform_fit_with_lens__source_galaxy(
    lens_galaxy=lens_galaxy, source_galaxy=source_galaxy
)

aplt.fit_imaging.subplot_fit_imaging(fit=correct_fit, include=aplt.Include(mask=True))

print("Bayesian Evidence of Incorrect Fit:")
print(fit.evidence)
print("Bayesian Evidence of Correct Fit:")
print(correct_fit.evidence)

# The evidence *is* lower. However, the difference in evidence isn't *that large*. This is going to be a problem for our
# non-linear search, as its going to see *a lot* of solutions with really high evidences. Furthermore, these solutions
# occupy the *vast majority* of parameter space (e.g. every single lens model that is wrong). This makes it easy for
# the non-linear search to get lost searching through these unphysical solutions and, unfortunately, infer an incorrect
# lens model (e.g. a local maxima).

# There is no simple fix for this. The reality is that for an inversion these solutions exist. This is why pipelines
# were initially conceived - as they offer a simple solution to this problem. We build a pipelin that begins by modeling
# the source galaxy as a light profile, 'initializing' our lens mass model. Then, when we switch to an inversion in the
# next phase, our mass model starts in the correct regions of parameter space and doesn'tget lost sampling these incorrect solutions.
#
# Its not ideal, but its also not a big problem. Furthermore, light-profiles run faster computationally than inversions,
# so we'd probably want to do this anyway!


# Okay, so we've covered incorrect solutions, lets end by noting that we can model profiles and inversions at the same
# time. We do this when we want to simultaneously fit and subtract the light of a lens galaxy and reconstruct its lensed
# source using an inversion. To do this, all we have to do is give the lens galaxy a light profile.


def simulate_lens_with_light_profile():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=0.8, sersic_index=4.0
        ),
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.1, 0.1),
            axis_ratio=0.8,
            phi=90.0,
            intensity=0.2,
            effective_radius=0.3,
            sersic_index=1.0,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(180, 180),
        pixel_scales=0.05,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


# When fitting such an image we now want to include the lens's light in the analysis. First, we update our
# mask to be circular so that it includes the central regions of the image and lens galaxy.
imaging = simulate_lens_with_light_profile()

mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=2.5
)

# As I said above, performing this fit is the same as usual, we just give the lens galaxy a light profile.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.SphericalSersic(
        centre=(0.0, 0.0), intensity=0.2, effective_radius=0.8, sersic_index=4.0
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
    ),
)

# These are all the usual things we do when setting up a fit.
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# This fit now subtracts the lens galaxy's light from the image and fits the resulting source-only image with the
# inversion. When we plotters the image, a new panel on the sub-plotters appears showing the model image of the lens galaxy.
fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

aplt.fit_imaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))

# Of course if the lens subtraction is rubbish so is our fit, so we can be sure that our lens model wants to fit the
# lens galaxy's light accurately (below, I've increased the lens galaxy intensity from 0.2 to 0.3).
lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.SphericalSersic(
        centre=(0.0, 0.0), intensity=0.3, effective_radius=0.8, sersic_index=4.0
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.fit(masked_dataset=masked_imaging, tracer=tracer)

aplt.fit_imaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))

# And with that, we're done. Finally, I'll point out a few things about what we've covered to get you thinking about
# the next tutorial on adaption.

# - The unphysical solutions above are clearly problematic. Whilst they have lower Bayesian evidences their existance
#   will still impact our inferred lens model. However, the pixelization's that we used in this chapter are not adapted
#   to the images they are fitting and this means that the correct solutions achieve much lower Bayesian evidence values
#   than is actually possible. Thus, once we've covered adaption, these issues will be resolved!

# - When the lens galaxy's light is subtracted perfectly it leaves no residuals. However, if it isn't subtracted
#   perfectly it does leave residuals and these residuals will be fitted by the inversion. If the residual are
#   significant this is going to mess with our source reconstruction and can lead to some pretty nasty
#   systematics. In the next chapter, we'll learn how our adaptive analysis can prevent this residual fitting.
