import autolens as al
import autolens.plot as aplt

# So, we can use an inversion to reconstruct an image. Furthermore, this reconstruction provides the 'best-fit'
# solution. And, when we inspect the fit with the fitting module, we see residuals indicative of a good fit.

# Everything sounds pretty good, doesn't it? You're probably thinking, why are there more tutorials? We can use
# inversions now, don't ruin it! Well, there is a problem - which I hid from you in the last tutorial, which we'll
# cover now.

# Lets use the same simple source as last time.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
        ),
    )

    source_galaxy_0 = al.Galaxy(
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

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy_0])

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


# We're going to perform a lot of fits using an inversion this tutorial. This would create a lot of code, so to keep
# things tidy, I've setup this function which handles it all for us.

# (You may notice we include an option to 'use_inversion_border, ignore this for now, as we'll be covering borders in
# the next tutorial)
def perform_fit_with_source_galaxy(source_galaxy):

    imaging = simulate()

    mask = al.mask.circular_annular(
        shape_2d=imaging.shape_2d,
        pixel_scales=imaging.pixel_scales,
        inner_radius=0.5,
        outer_radius=2.2,
    )

    masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.fit(masked_dataset=masked_imaging, tracer=tracer)


# Okay, so lets look at our fit from the previous tutorial in more detail. We'll use a higher resolution 40 x 40 grid.
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

fit = perform_fit_with_source_galaxy(source_galaxy=source_galaxy)

aplt.fit_imaging.subplot_fit_imaging(fit=fit)

# It looks pretty good! However, this is because I sneakily chose a regularization coefficient that gives a good looking
# solution. If we reduce this regularization coefficient to zero, our source reconstruction goes weird.
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=0.0),
)

no_regularization_fit = perform_fit_with_source_galaxy(source_galaxy=source_galaxy)

aplt.fit_imaging.subplot_fit_imaging(
    fit=no_regularization_fit, include=aplt.Include(mask=True)
)

# So, what's happening here? Why does reducing the regularization do this to our source reconstruction?

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

# If we change the 'normalization' variables of the plotter such that the color-map is restricted to a narrower
# range of values, we can see that even without regularization we are still reconstructing the actual source galaxy.
aplt.inversion.reconstruction(
    inversion=no_regularization_fit.inversion,
    plotter=aplt.Plotter(cmap=aplt.ColorMap(norm_max=0.5, norm_min=-0.5)),
)

# Over-fitting is why regularization is necessary. Solutions like this completely ruin our attempts to model a
# strong lens. By smoothing our source reconstruction we ensure it doesn't fit the noise in the image. If we set a
# really high regularization coefficient we completely remove over-fitting at the expense of also fitting the
# image less accurately.
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=100.0),
)

high_regularization_fit = perform_fit_with_source_galaxy(source_galaxy=source_galaxy)

aplt.fit_imaging.subplot_fit_imaging(
    fit=high_regularization_fit, include=aplt.Include(mask=True)
)

# So, we now understand regularization and its purpose. But there is one nagging question that remains, how do I choose
# the regularization coefficient? We can't use our likelihood, as decreasing the regularization coefficient will always
# increase the likelihood, because it allows the source reconstruction to fit the dataset better.
print("Likelihood Without Regularization:")
print(no_regularization_fit.likelihood_with_regularization)
print("Likelihood With Normal Regularization:")
print(fit.likelihood_with_regularization)
print("Likelihood With High Regularization:")
print(high_regularization_fit.likelihood_with_regularization)

# If we used the likelihood we will always choose a coefficient of 0! We need a different goodness-of-fit measure. For
# this, we invoke the 'Bayesian evidence', which quantifies the goodness of the fit as follows:

# - First, it requires that the residuals of the fit are consistent with Gaussian noise (which is the noise expected
#   in imaging). If this Gaussian pattern is not visible in the residuals, it tells us that the noise must have been
#   over-fitted. Thus, the Bayesian evidence decreases. Obviously, if the image is poorly fitted, the residuals don't
#   appear Gaussian either, but the poor fit will lead to a decrease in Bayesian evidence decreases all the same!

# - This leaves us with a large number of solutions which all fit the dataset equally well (e.g., to the noise level). To
#   determine the best-fit from these solutions the Bayesian evidence quantifies the complexity of each solution's
#   source reconstruction. If the inversion requires lots of pixels and a low level of regularization to achieve a good
#   fit, the Bayesian evidence decreases. It penalizes solutions which are complex, which, in a Bayesian sense, are less
#   probable (you may want to look up 'Occam's Razor').

# If a really complex source reconstruction is paramount to fitting the image accurately than that is probably the
# correct solution. However, the Bayesian evidence ensures we only invoke this more complex solution when the dataset
# necessitates it.

# Lets take a look at the Bayesian evidence:
print("Bayesian Evidence Without Regularization:")
print(no_regularization_fit.evidence)
print("Bayesian Evidence With Normal Regularization:")
print(fit.evidence)
print("Bayesian Evidence With High Regularization:")
print(high_regularization_fit.evidence)

# Great! As expected, the solution that we could see 'by-eye' was the best solution corresponds to
# the highest evidence solution.

# Before we end, lets consider which aspects of an inversion are linear and which are non-linear.

# The linear part of the linear inversion solves for the 'best-fit' solution. For a given regularizaton coefficient,
# this includes the regularization pattern. That is, we linearly reconstruct the combination of source-pixel fluxes
# that best-fit the image *including* the penalty term due to comparing neighboring source-pixel fluxes.

# However, determining the regularization coefficient that maximizes the Bayesian evidence remains a non-linear
# problem and this becomes part of our non-linear search. The Bayesian evidence also depends on the source
# resolution which means the pixel-grid resolution may also be part of our non-linear search. Nevertheless,
# this is only 3 parameters - there were 30+ when using light profiles to represent the source!

# Here are a few questions for you to think about.

# 1) We maximize the evidence by using simpler source reconstructions. Therefore, decreasing the pixel-grid size should
#    provide a higher evidence, provided it still has enough resolution to fit the image well (and provided that the
#    regularization coefficient is still an appropriate value). Can you increase the evidence from the value above by
#    changing these parameters - I've set you up with a code to do so below.
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

fit = perform_fit_with_source_galaxy(source_galaxy=source_galaxy)

print("Previous Bayesian Evidence:")
print(10395.370224426646)
print("New Bayesian Evidence:")
print(fit.evidence)

aplt.fit_imaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))

# 2) Can you think of any other ways we might increase the evidence even further? If not - don't worry about - but
#    you'll learn that PyAutoLens actually adapts its source reconstructions to the properties of the image that it is
#    fitting, so as to objectively maximize the evidence!
