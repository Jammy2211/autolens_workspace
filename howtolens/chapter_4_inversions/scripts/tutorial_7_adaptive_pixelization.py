import autolens as al
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.model.inversion.plotters import inversion_plotters

# In this tutorial we'll use a new pixelization, called the VoronoiMagnification pixelization. This pixelization
# doesn't use a uniform grid of rectangular pixels, but instead uses an irregular 'Voronoi' pixels.
# So, why do we want to do that? Lets take another quick look at the rectangular grid..

# This simulates the same the image we've fitted in the past few tutorials.
def simulate():

    psf = al.PSF.from_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(150, 150), pixel_scale=0.05, sub_grid_size=2
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mass_profiles.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.7,
            phi=135.0,
            intensity=0.2,
            effective_radius=0.2,
            sersic_index=2.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.05,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=1.0,
        add_noise=True,
    )


# Lets quickly remind ourselves of the image and the 3.0" circular mask we'll use to mask it.
ccd_data = simulate()

mask = al.Mask.circular(
    shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=2.5
)

al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data, mask=mask)

# The lines of code below do everything we're used to, that is, setup an image, mask it, trace it via a tracer,
# setup the rectangular mapper, etc.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
    ),
)

lens_data = al.LensData(ccd_data=ccd_data, mask=mask)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pix.Rectangular(shape=(40, 40)),
    regularization=reg.Constant(coefficient=0.5),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

al.lens_fit_plotters.plot_fit_subplot(
    fit=fit, should_plot_mask=True, extract_array_from_mask=True, zoom_around_mask=True
)

# Okay, so lets think about the rectangular pixelization. Is this the optimal way to reconstruct our source? Are there
# features in the source-plane that arn't ideal? How do you think we could do a better job?

# Well, given we're doing a whole tutorial on using a different pixelization to the rectangular grid, you've probably
# guessed that it isn't optimal. Infact, its pretty rubbish, and not a pixelization we should actually want to model
# many lenses with!

# So what is wrong with the grid? Well, lets think about the source reconstruction.
inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# There is one clear problem, we are using a small number of the total source pixels to reconstruct the source. The
# majority of source pixels are located away from the source. By my estimate, we're using just 16 pixels (the
# central 4x4 grid) out of the 1600 pixels to actually fit the data! The remaining ~1500 pixels are doing *nothing*
# but fit noise.

# This is a waste and our analysis will take longer to run because of it. However, more importantly, it means that our
# Bayesian regularization scheme is sub-optimal. In tutorial 4, we discussed how the Bayesian evidence of the
# regularization wants to obtain the *simplest* source solution possible. That is the solution which fits the data well
# using the fewest source pixels. Clearly, if we dedicating a huge number of source pixels to doing *nothing*, our source
# reconstruction will be unecessarily complex (and therefore lower evidence).

# If our pixelization could 'focus' its pixels where we actually have more data, e.g. the highly magnified regions
# of the source-plane, we could reconstruct the source using far fewer pixels. That'd be great both for computational
# efficiency and increasing the Bayesian evidence and that is exactly what our Voronoi grid does.

# To achieve this, we first compute an 'image-plane pixelization grid', which is a set of sparse coordinates in the
# image-plane that will be ray-traced to the source-plane and define the centres of our source-pixel grid. We compute
# this grid directly from a pixelization, by passing it a grid.

adaptive = pix.VoronoiMagnification(shape=(20, 20))

pixelization_grid = adaptive.pixelization_grid_from_grid(grid=lens_data.grid)

# We can plot this grid over the image, to see that it is a coarse grid over-laying the image itself.

al.ccd_plotters.plot_image(
    ccd_data=ccd_data,
    grid=pixelization_grid,
    mask=mask,
    zoom_around_mask=True,
    extract_array_from_mask=True,
)

# When we pass a tracer a source galaxy with this pixelization it automatically computes the ray-traced source-plane
# Voronoi grid using the grid above. Thus, our Voronoi pixelization is used by the tracer's fit.

source_galaxy = al.Galaxy(
    redshift=1.0, pixelization=adaptive, regularization=reg.Constant(coefficient=1.0)
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# If we look at the lens fit, we'll that our source-plane no longer uses rectangular pixels, but Voronoi pixels!

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

al.lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# And we can take a closer inspection of the inversion itself.
inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# Clearly, this is an improvement. We're using fewer pixels than the rectangular grid (400, instead of 1600), but
# reconstructing our source is far greater detail. A win all around? It sure is.

# On our rectangular grid, we regularized each source pixel with its 4 neighbors. We compared their fluxes, summed the
# differences, and penalized solutions where the differences were large. For a Voronoi grid, we do the same thing,
# now comparing each source-pixel with all other source-pixels with which it shares a direct vertex. This means that
# different source-pixels may be regularized with different numbers of source-pixels, depending on how many neighbors
# are formed.

# This Voronoi magnification grid is still far from optimal. There are lots of source-pixels effectively fitting just
# noise. We may achieve even better solutions if the central regions of the source were reconstructed using even more
# pixels. So, how do we improve on this? Well, you'll have to wait until chapter 5, when we introduce PyAutoLens's
# adaptive functionality, or 'hyper-mode'.

# In the mean time, you may wish to experiment with using both Rectangular and VoronoiMagnification grids to fit
# lenses which can be easily achieve by changing the input pixeliation given to a pipeline.
