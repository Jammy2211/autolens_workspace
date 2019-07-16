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
from autolens.model.inversion.plotters import inversion_plotters
from autolens.lens.plotters import lens_fit_plotters
from autolens.plotters import array_plotters

# This chapter is called 'hyper_mode'. So, that begs the question, what is hyper mode?
#
# Well, in hyper-mode, we use previous models of a strong gravitational lens to improve and adapt subsequent fits.
# That's exactly what we did in the tutorial, when we defined a hyper-image as follows

# hyper_image = fit_compact.model_image(return_in_2d=False)

# Although I didn't explain how, this hyper-image was used to adapt our pixelization to the compact source's
# morphology, which we saw gave a much better fit to the data than a magnification based grid!

# Now, we'll go into the details of how this works. To do this, we'll use the image of the same compact source galaxy
# as before and we'll set up the hyper-image like we did in the previous tutorial, by first fitting the image with a
# magnification based pixelization.

# This is the usual simulate function, using the compact source of the previous tutorial.


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

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy],
        source_galaxies=[source_galaxy],
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


# Lets simulate the data, draw a 3.0" mask and set up the lens data that we'll fit.

ccd_data = simulate()
mask = msk.Mask.circular(shape=(150, 150), pixel_scale=0.05, radius_arcsec=3.0)
lens_data = ld.LensData(ccd_data=ccd_data, mask=mask)

# Next, we're going to fit the image using our magnification based grid. The code below does all the usual steps
# required to do this.

lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
    ),
)

source_weight_power_10 = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiMagnification(shape=(30, 30)),
    regularization=reg.Constant(coefficient=3.3),
)

pixelization_grid = source_weight_power_10.pixelization.pixelization_grid_from_grid_stack(
    grid_stack=lens_data.grid_stack
)

grid_stack_with_pixelization_grid = lens_data.grid_stack.new_grid_stack_with_grids_added(
    pixelization=pixelization_grid
)

tracer = ray_tracing.TracerImageSourcePlanes(
    lens_galaxies=[lens_galaxy],
    source_galaxies=[source_weight_power_10],
    image_plane_grid_stack=grid_stack_with_pixelization_grid,
    border=lens_data.border,
)

fit = lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

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

# And, after all that, we can set up our hyper image again!

hyper_image = fit.model_image(return_in_2d=False)

# Okay, so how does this hyper_image let us adapt our source-plane pixelization? Well, we derive our source-plane
# pixelization using a 'weighted KMeans clustering algorithm', which is a standard algorithm for partioning data in
# statistics.
#
# In simple terms, this algorithm works as follows:

# 1) Give the KMeans algorithm a set of weighted data (e.g. the hyper image).

# 2) For a given number of clusters, this algorithm will find a set of (y,x) coordinates or 'clusters' that are equally
#    distributed over the weighted data. Where the data is weighted higher, more clusters will congregate, and visa
#    versa.

# 3) The returned (y,x) 'clusters' then make up our source-pixel centres and, as described, there will be more
#    wherever our hyper-image is brighter! Like we did for the magnification based pixelization, we can then trace
#    these coordinates to the source-plane to define our source-pixel grid.

# This is a fairly simplistic description of a KMeans algorithm. Feel free to check out the links below for a more
# in-depth view:

# https://en.wikipedia.org/wiki/K-means_clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

# Okay, so we now have a sense of how our VoronoiBrightness pixelization is drawn. Now, considering how we create
# the weighted data the KMeans algorithm clusters.

# This image, called the 'cluster_weight_map'  is generated using the 'weight_floor' and 'weight_power' parameters
# of the VoronoiBrightness pixelization. The cluster weight map is generated following 4 steps:

# 1) Increase all values of the hyper-image that are < 0.02 to 0.02. This is necessary because negative values and
#    zeros break the KMeans clustering algorithm.

# 2) Divide all values of this image by its maximum value, such that the hyper-image now only contains values between
#    0.0 and 1.0 (where the values of 1.0 were the maximum values of the hyper-image)

# 3) Add the weight_floor to all values (a weight_floor of 0.0 therefore does not change the cluster weight map)

# 4) Raise all values to the power of weight_power (a weight_power of 1.0 therefore does not change the cluster weight
#    map, whereas a value of 0.0 means all values 1.0 and therefore weighted equally).

# Lets look at this in action. We'll first inspect 3 cluster_weight_maps for a weight_power of 0.0, 5.0 and 10.0,
# setting the weight_floor to 0.0 such that it does not change the cluster weight map.

source_weight_power_0 = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=0.0
    ),
    regularization=reg.Constant(coefficient=1.0),
)

cluster_weight_power_0 = source_weight_power_0.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=hyper_image
)
cluster_weight_power_0 = mask.scaled_array_2d_from_array_1d(
    array_1d=cluster_weight_power_0
)
array_plotters.plot_array(
    array=cluster_weight_power_0,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

source_weight_power_5 = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=reg.Constant(coefficient=1.0),
)

cluster_weight_power_5 = source_weight_power_5.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=hyper_image
)
cluster_weight_power_5 = mask.scaled_array_2d_from_array_1d(
    array_1d=cluster_weight_power_5
)
array_plotters.plot_array(
    array=cluster_weight_power_5,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

source_weight_power_10 = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=10.0
    ),
    regularization=reg.Constant(coefficient=1.0),
)

cluster_weight_power_10 = source_weight_power_10.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=hyper_image
)
cluster_weight_power_10 = mask.scaled_array_2d_from_array_1d(
    array_1d=cluster_weight_power_10
)
array_plotters.plot_array(
    array=cluster_weight_power_10,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# So, as expected, when we increased the weight-power the brightest regions of the hyper-image become weighted higher
# relative to the fainter regions. This is exactly what we want, as it means the KMeans algorithm will adapt its
# pixelization to the brightest regioons of the source.

# Lets check the resulting fits using each cluster weight map. We'll define a convenience function to for this:


def fit_lens_data_with_source_galaxy(lens_data, source_galaxy):

    pixelization_grid = source_galaxy.pixelization.pixelization_grid_from_grid_stack(
        grid_stack=lens_data.grid_stack,
        hyper_image=hyper_image,
        cluster=lens_data.cluster,
    )

    grid_stack_with_pixelization_grid = lens_data.grid_stack.new_grid_stack_with_grids_added(
        pixelization=pixelization_grid
    )

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy],
        source_galaxies=[source_galaxy],
        image_plane_grid_stack=grid_stack_with_pixelization_grid,
        border=lens_data.border,
    )

    return lens_fit.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)


fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_weight_power_0
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_weight_power_5
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_weight_power_10
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# Great, so this is why our pixelization can adapt to the source's brightness.
#
# So, what is the weight_floor for? The weight-power congregates pixels around the source. However, there is a risk
# that by congregating too many source pixels in its brightest regions, we lose resolution further out, where the
# source is bright, but not its brightest!

# The noise-floor allows these regions to maintain a higher weighting whilst using the noise_power increases, such
# that the pixelization can fully adapt to the source's brightest and faintest regions.

# Lets look at once example:

source_weight_floor = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.5, weight_power=10.0
    ),
    regularization=reg.Constant(coefficient=1.0),
)

cluster_weight_floor = source_weight_floor.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=hyper_image
)
cluster_weight_floor = mask.scaled_array_2d_from_array_1d(array_1d=cluster_weight_floor)
array_plotters.plot_array(
    array=cluster_weight_floor,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_weight_floor
)

inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# And that is how the first feature in hyper-mode works. Now, lets think about the Bayesian evidence, which we
# noted in the previous chapter went to significantly higher values than the magnification-based gird. In chapter 4
# we described how the Bayesian evidence quantities the overall goodness-of-fit:

# - First, it requires that the residuals of the fit are consistent with Gaussian noise (which is the noise expected
#   in CCD imaging). If this Gaussian pattern is not visible in the residuals, it tells us that the noise must have been
#   over-fitted. Thus, the Bayesian evidence decreases. Obviously, if the image is poorly fitted, the residuals don't
#   appear Gaussian either, but the poor fit will lead to a decrease in Bayesian evidence decreases all the same!

# - This leaves us with a large number of solutions which all fit the data equally well (e.g., to the noise level). To
#   determine the best-fit from these solutions, the Bayesian evidence quantifies the complexity of each solution's
#   source reconstruction. If the inversion requires lots of pixels and a low level of regularization to achieve a good
#   fit, the Bayesian evidence decreases. It penalizes solutions which are complex, which, in a Bayesian sense, are less
#   probable (you may want to look up 'Occam's Razor').

# Can you think why adapting to the source's brightness increases the evidence?

# Its because, by adapting to the source's morphology, we can now access solutions that fit the data really well
# (e.g. to the Gaussian noise-limit) but use significantly fewer source-pixels than other grids. For instance, a
# typical magnification based grid uses resolutions of 40 x 40, or 1600 pixels. In contrast, a morphology based grid
# typically uses just 300-800 pixels (depending on the source itself). Clearly, the easiest way to make our source
# solution simpler is to use fewer pixels overall!

# This provides a second benefit. If the best solutions in our fit want to use the fewest source-pixels possible, and
# PyAutoLens can now access those solutions, this means that hyper-mode will run much faster than the
# magnification based grid! Put simply, fewer source-pixels means lower computational overheads. YAY!

# Finally, its worth pointing out that hyper-mode is another reason why PyAutoLens uses the pipeline framework.
# Clearly, to set up a hyper-image we need a previous models of a lens to be accessible. In tutorial 5, I'll
# demonstrate how pipelines have a variety of built-in features that make passing hyper-images through the pipeliine
# seamless.
