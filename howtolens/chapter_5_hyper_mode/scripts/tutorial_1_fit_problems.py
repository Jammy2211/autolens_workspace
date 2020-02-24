import autolens as al
import autolens.plot as aplt

# To begin, make sure you have read the 'introduction' file carefully, as a clear understanding of
# how the Bayesian evidence works is key to understanding this chapter!

# In the previous chapter we investigated two pixelizations; Rectanguar and VoronoiMagnification. We learnt that the
# latter was better than the former, because it dedicated more source-pixels to the regions of the source-plane where
# we had more simulator, e.g, the high-magnification regions. Therefore, we could fit the dataset using fewer source pixels,
# which improved computational efficiency and the Bayesian evidence.

# So far, we've used just one regularization scheme; Constant. As the name suggests, this regularization scheme
# applies just one regularization coefficient when regularizing source-pixels with one another. In case you've forgot,
# here is a refresher of regularization, from chapter 4:

# -------------------------------------------- #

# When our inversion reconstructs a source, it doesn't *just* compute the set of fluxes that best-fit the image. It also
# 'regularizes' this solution, going to every pixel on our rectangular grid and comparing its reconstructed flux with
# its 4 neighboring pixels. If the difference in flux is large the solution is penalized, reducing its likelihood.
# You can think of this as us applying a prior that our source galaxy solution is 'smooth'.

# This adds a 'penalty term' to the likelihood of an inversion which is the summed difference between the
# reconstructed fluxes of every source-pixel pair multiplied by the regularization coefficient. By setting the
# regularization coefficient to zero, we set this penalty term to zero, meaning that regularization is omitted.

# Why do we need to regularize our solution? Well, we just saw why - if we don't apply this smoothing, we 'over-fit'
# the image. More specifically, we over-fit the noise in the image, which is what the large flux values located
# at the exteriors of the source reconstruction are doing. Think about it, if your sole aim is to maximize the
# likelihood, the best way to do this is to fit *everything* accurately, including the noise.

# ----------------------------------------------#

# So, when using a Constant regularization scheme, we regularize the source by adding up the difference in fluxes
# between all source-pixels multiplied by one single value of a regularization coefficient. This means that every
# single source pixel receives the same 'level' of regularization, regardless of whether it is reconstructing the
# bright central regions of the source or its faint exterior regions.


# In this tutorial, we'll learn that our magnification-based pixelization and constant regularization schemes are
# far from optimal. To understand why, we'll inspect fits to three strong lenses, simulated using the same mass
# profile but with different sources whose light profiles become gradually more compact. For all 3 fits, we'll use
# the same source-plane resolution and a regularization coefficient that maximize the Bayesian evidence. Thus, these
# are the 'best' source reconstructions we can hope to achieve when adapting to the magnification.

# We'll use 3 sources whose effective radius and Sersic index are changed such that each is more compact that the last.

source_galaxy_flat = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.7,
        phi=135.0,
        intensity=0.2,
        effective_radius=0.5,
        sersic_index=1.0,
    ),
)

source_galaxy_compact = al.Galaxy(
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

source_galaxy_super_compact = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.7,
        phi=135.0,
        intensity=0.2,
        effective_radius=0.1,
        sersic_index=4.0,
    ),
)

# The function below uses each source galaxy to simulate the imaging dataset. It performs the usual tasks we are used to seeing
# (make the PSF, galaxies, tracer, etc.).


def simulate_for_source_galaxy(source_galaxy):

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
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


# We'll use the same 3.0" mask to fit all three of our sources.
mask = al.mask.circular(shape_2d=(150, 150), pixel_scales=0.05, radius=3.0)

# Now, lets simulate all 3 of our source's as imaging dataset.

imaging_source_flat = simulate_for_source_galaxy(source_galaxy=source_galaxy_flat)

imaging_source_compact = simulate_for_source_galaxy(source_galaxy=source_galaxy_compact)

imaging_source_super_compact = simulate_for_source_galaxy(
    source_galaxy=source_galaxy_super_compact
)

# We'll make one more useful function which fits each simulated imaging with a VoronoiMagniication pixelization
# and Constant regularization scheme.

# We'll input the regularization coefficient of each fit, so that for each simulated source we regularize it at
# an appropriate level. Again, there is nothing new in this function you haven't seen before.


def fit_imaging_with_voronoi_magnification_pixelization(
    imaging, mask, regularization_coefficient
):

    masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        pixelization=al.pix.VoronoiMagnification(shape=(30, 30)),
        regularization=al.reg.Constant(coefficient=regularization_coefficient),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.fit(masked_dataset=masked_imaging, tracer=tracer)


# Lets fit our first source with the flattest light profile. One should note that this uses the highest
# regularization coefficient of our 3 fits (as determined by maximizing the Bayesian evidence).

fit_flat = fit_imaging_with_voronoi_magnification_pixelization(
    imaging=imaging_source_flat, mask=mask, regularization_coefficient=9.2
)

aplt.fit_imaging.subplot_fit_imaging(
    fit=fit_flat,
    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
)

aplt.inversion.reconstruction(
    inversion=fit_flat.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

print(fit_flat.evidence)

# Okay, so the fit was *excellent*. There were effectively no residuals in the fit, and the source has been
# reconstructed using lots of pixels! Nice!

# Now, lets fit the next source, which is more compact.

fit_compact = fit_imaging_with_voronoi_magnification_pixelization(
    imaging=imaging_source_compact, mask=mask, regularization_coefficient=3.3
)

aplt.fit_imaging.subplot_fit_imaging(
    fit=fit_compact,
    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
)

aplt.inversion.reconstruction(
    inversion=fit_compact.inversion,
    include=aplt.Include(inversion_pixelization_grid=True),
)

print(fit_compact.evidence)

# Oh no! The fit doesn't look so good! Sure, we reconstruct *most* of the lensed source's structure, but there are two
# clear 'blobs' in the residual map where we are failing to reconstruct the central regions of the source galaxy.

# Take a second to think about why this might be. Is it the pixelization? The regularization?

# Okay, so finally, we're going to fit our super compact source. Given that the results for the compact source didn't
# look so good, you'd be right in assuming this is just going to make things even worse. Again, think about why this
# might be.

fit_super_compact = fit_imaging_with_voronoi_magnification_pixelization(
    imaging=imaging_source_super_compact, mask=mask, regularization_coefficient=3.1
)

aplt.fit_imaging.subplot_fit_imaging(
    fit=fit_super_compact,
    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
)

aplt.inversion.reconstruction(
    inversion=fit_super_compact.inversion,
    include=aplt.Include(inversion_pixelization_grid=True),
)

print(fit_super_compact.evidence)

# Okay, so what did we learn? The more compact our source, the worse the fit. This happens even though we are using the
# *correct* lens mass model, telling us that something is going fundamentally wrong with our source reconstruction and
# inversion. As you might of guessed, both our pixelization and regularization scheme are to blame, dammit!

# __Pixelization__

# For the pixelization the problem is the same one we found when comparing the Rectangular and VoronoiMagnification
# pixelizations. Put simply, we are not dedicating enough source-pixels to the central regions of the source
# reconstruction, e.g. where it's brightest. As the source becomes more compact, the source reconstruction doesn't have
# enough resolution to resolve its fine-detailed central structure, causing the fit to the image to degrade.

# Think about it, as we made our sources more compact we go from reconstructing them using ~100 source pixels,
# to ~20 source pixels to ~ 10 source pixels. This is why we advocated not using the Rectangular pixelization in the
# previous chapter!

# It turns out that adapting to the magnification wasn't the best idea all along. As we simulated more compact sources
# the magnification (which is determined via the mass model) didn't change. So, we foolishly reconstructed each source
# using fewer and fewer pixels, leading to a worse and worse fit! Furthermore, these source's happened to be located
# in the highest magnification regions of the source plane! If the source's were further away from the centre of the
# caustic, the VoronoiMagnification pixelization would use *even less* pixels to reconstruct it. That is NOT what we
# want!

# __Regularization__

# Regularization also causes problems. When using a Constant regularization scheme, we regularize the source by
# adding up the difference in fluxes between all source-pixels multiplied by one single value of a regularization
# coefficient. This means that, every single source pixel receives the same 'level' of regularization, regardless of
# whether it is reconstructing the bright central regions of the source or its faint exterior regions. Lets look:

aplt.inversion.regularization_weights(
    inversion=fit_compact.inversion,
    include=aplt.Include(inversion_pixelization_grid=True),
)

# As you can see, all pixels are regularized with our input regularization coefficient value of 3.6.

# Is this the best way to regularize the source? Well, as you've probably guessed, it isn't. But why not? Its
# because different regions of the source demand different levels of regularization:

# 1) In the source's central regions its flux gradient is steepest; the change in flux between two source pixels is
#    much larger than in the exterior regions where the gradient is flatter (or there is no source flux at all). To
#    reconstruct the detailed structure of the source's cuspy inner regions, the regularization coefficient needs to
#    be much lower to avoid over-smoothing.

# 2) On the flip side, the source reconstruction wants to assume a high regularization coefficient further out
#    because the source's flux gradient is flat (or there is no source signal at all). Higher regularization
#    coefficients will increase the Bayesian evidence because by smoothing more source-pixels it makes the solution
#    'simpler', given that correlating the flux in these source pixels the solution effectively uses fewer
#    source-pixels (e.g. degrees of freedom).

# So, herein lies the pitfall of a constant regularization scheme. Some parts of the reconstructed source demand a low
# regularization coefficient whereas other parts want a high value. Unfortunately, we end up with an intermediate
# regularization coefficient that over-smooths the source's central regions whilst failing to fully correlate exterior
# pixels. Thus, by using an adaptive regularization scheme, new solutions that further increase the Bayesian evidence
# become accessible.


# __Noise Map__

# Before we wrap up this tutorial, I want us to also consider the role of our noise-map and get you thinking about why
# we might want to scale its variances. Lets look at the super-compact fit again;

aplt.fit_imaging.subplot_fit_imaging(
    fit=fit_super_compact,
    include=aplt.Include(inversion_image_pixelization_grid=True, mask=True),
)

# So, whats the problem? Look closely at the 'chi-squared image'. Here, you'll note that a small subset of our dataset
# have extremely large chi-squared values. This means our non-linear search (which is trying minimize chi-squared)
# is going to seek solutions which primarily only reduce these chi-squared values. For the image above a small subset
# of the dataset (e.g. < 5% of pixels) contributes to the majority of the likelihood (e.g. > 95% of the overall chi-squared).
# This is *not* what we want, as instead of using the entire surface brightness profile of the lensed source galaxy to
# fit our lens model, we end up using only a small subset of its brightest pixels.

# In the context of the Bayesian evidence things become even more problematic. The Bayesian evidence is trying to
# achieve a well-defined solution; a solution that provides a reduced chi-squared of 1. This solution is poorly defined
# when the chi-squared image looks like the one above. When a subset of pixels have chi-squareds > 300, the only way to
# achieve a reduced chi-squared 1 is to reduce the chi-squareds of other pixels to 0, e.g. by over-fitting their noise.
# Thus, we quickly end up in a regime where the choice of regularization coefficient is ill defined.

# With that, we have motivated hyper_galaxy-mode. To put it simply, if we don't adapt our pixelizations, regularization and
# noise-map, we will get solutions which reconstruct the source poorly, regularize the source sub-optimally and over-fit a
# small sub-set of image pixels. Clearly, we want an adaptive pixelization, regularization scheme and noise-map,
# which what we'll cover tutorials 2, 3 and 4!
