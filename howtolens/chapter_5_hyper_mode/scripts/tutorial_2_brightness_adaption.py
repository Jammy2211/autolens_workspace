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
from autolens.plotters import array_plotters

# So, in the previous tutorial, we motivated a need to adapt our pixelization to the source's morphology, that is where
# it congregates pixels in the source's brightest regions, regardless of where it is located in the source-plane. This
# raises an interesting question; how do we adapt our source pixelization to the reconstructed source, before we've
# actually reconstructed the source and therefore know what to adapt it too?

# To do this, we define a 'hyper-image' of the lensed source galaxy, which is a model image of the source computed
# using a previous lens model fit to the image (e.g. in an earlier phase of a pipeline). Because this image tells us
# where in the image our source is located, it means that by tracing these pixels to the source-plane we can use them
# to tell us where we need to adapt our source pixelization!

# So, lets go into the details of how this works. We'll use the same compact source galaxy as the previous tutorial and
# we'll begin by fitting it with a magnification based pixelizationo. Why? So we can use its model image to set up
# the hyper-image.

# This is the usual simulate function, using the compact source of the previous tutorial.


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

# Next, we're going to fit the image using our magnification based grid. The code below does all the usual steps
# required to do this.

lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
    ),
)

source_magnification = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiMagnification(shape=(30, 30)),
    regularization=reg.Constant(coefficient=3.3),
)


# This convenience function fits the image using a source-galaxy (which may have a hyper-image attatched to it).


def fit_lens_data_with_source_galaxy(lens_data, source_galaxy):

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


# Great, so now lets fit the instrument using our magnification based pixelizatioin and have a quick look to make sure it
# has the same residuals we saw in tutorial 1.

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

# Finally, we can use this fit to set up our hyper-image. Sure, the hyper-image isn't perfect, after all the were clear
# residuals in the central regions of the reconstructed source. But it's *okay*, it'll certainly gives us enough
# information on where the lensed source is located to adapt our pixelization.
hyper_image_1d = fit.model_image(return_in_2d=False)

# Okay, so now lets take a look at our brightness based adaption in action! Below, we define a source-galaxy using
# our new 'VoronoiBrightnessImage' pixelization and use this to fit the lens-instrument. One should note that we also
# attach the hyper-image to this galaxy. This is because it's pixelization uses this hyper-image to adapt, thus the
# galaxy needs to know what hyper-image it should use!

source_brightness = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=10.0
    ),
    regularization=reg.Constant(coefficient=0.5),
    hyper_galaxy_image_1d=hyper_image_1d,
)

fit = fit_lens_data_with_source_galaxy(
    lens_data=lens_data, source_galaxy=source_brightness
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

# Would you look at that! Our reconstruction of the image no longer has residuals! By congregating more source pixels
# in the brightest regions of the source reconstruction, we got a better fit. Furthermore, we can check that this
# indeed corresponds to an increase in Bayesian evidence, noting that the evidence of the compact source when
# using a VoronoiMagnification pixelization was 14236:

print("Evidence using magnification based pixelization = ", 14236.292117135737)

print("Evidence using brightness based pixelization = ", fit.evidence)


# It increases! By over 200, which, for a Bayesian evidence, is pretty damn large! Clearly, By any measure, this
# pixelization is a huge success. It turns out that, all along, we should have been adapting to the source's brightness!
# In doing so, we will *always* reconstruct the detailed structure of the source's brightest regions with a sufficiently
# high resolution. Hurrah!

# Okay, so we can now adapt our pixelization to the morphology of our lensed source galaxy. To my knowledge,
# this is the *best* approach one can take the lens modeling. Its more tricky to implement (as I'll explain next)
# and introduces a few extra hyper-parameters that we'll fit for. But, the pay-off is more than worth it, as we fit
# our imaging data better and (typically) end up using far fewer source pixels to fit the instrument, as we don't 'waste'
# pixels reconstructing regions of the source-plane where there is no signal.


# Okay, so how does the hyper_image actually adapt our pixelization to the source's brightness? We derive the
# pixelization using a 'weighted KMeans clustering algorithm', which is a standard algorithm for partioning instrument in
# statistics.

# In simple terms, this algorithm works as follows:

# 1) Give the KMeans algorithm a set of weighted instrument (e.g. determined from the hyper image).

# 2) For a given number of clusters, this algorithm will find a set of (y,x) coordinates or 'clusters' that are equally
#    distributed over the weighted instrument. Where the instrument is weighted higher, more clusters will congregate, and visa
#    versa.

# 3) The returned (y,x) 'clusters' then make up our source-pixel centres and, as described, there will be more
#    wherever our hyper-image is brighter! Like we did for the magnification based pixelization, we can then trace
#    these coordinates to the source-plane to define our source-pixel grid.

# This is a fairly simplistic description of a KMeans algorithm. Feel free to check out the links below for a more
# in-depth view:

# https://en.wikipedia.org/wiki/K-means_clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# Okay, so we now have a sense of how our VoronoiBrightnessImage pixelization is computed. Now, lets look at how we
# create the weighted instrument the KMeans algorithm clusters.

# This image, called the 'cluster_weight_map'  is generated using the 'weight_floor' and 'weight_power' parameters
# of the VoronoiBrightness pixelization. The cluster weight map is generated following 4 steps:

# 1) Increase all values of the hyper-image that are < 0.02 to 0.02. This is necessary because negative values and
#    zeros break the KMeans clustering algorithm.

# 2) Divide all values of this image by its maximum value, such that the hyper-image now only contains values between
#    0.0 and 1.0 (where the values of 1.0 were the maximum values of the hyper-image)

# 3) Add the weight_floor to all values (a weight_floor of 0.0 therefore does not change the cluster weight map)

# 4) Raise all values to the power of weight_power (a weight_power of 1.0 therefore does not change the cluster weight
#    map, whereas a value of 0.0 means all values 1.0 and therefore weighted equally).

# Lets look at this in action. We'll inspect 3 cluster_weight_maps, using a weight_power of 0.0, 5.0 and 10.0,
# setting the weight_floor to 0.0 such that it does not change the cluster weight map.

source_weight_power_0 = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=0.0
    ),
    regularization=reg.Constant(coefficient=1.0),
    hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_power_0 = source_weight_power_0.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_power_0.hyper_galaxy_image_1d
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
    hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_power_5 = source_weight_power_5.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_power_5.hyper_galaxy_image_1d
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
    hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_power_10 = source_weight_power_10.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_power_10.hyper_galaxy_image_1d
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
# pixelization to the brightest regions of the source.


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

# So, what does the weight_floor do? Increasing the weight-power congregates pixels around the source. However,
# there is a risk that by congregating too many source pixels in its brightest regions, we lose resolution further out,
# where the source is bright, but not its brightest!

# The noise-floor allows these regions to maintain a higher weighting whilst the noise_power increases. This means
# that the pixelization can fully adapt to the source's brightest and faintest regions simultaneously.

# Lets look at once example:

source_weight_floor = g.Galaxy(
    redshift=1.0,
    pixelization=pix.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.5, weight_power=10.0
    ),
    regularization=reg.Constant(coefficient=1.0),
    hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_floor = source_weight_floor.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_floor.hyper_galaxy_image_1d
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

# And that is how the first feature in hyper-mode works.

# Finally, lets think about the Bayesian evidence, which we have noted went to significantly higher values than the
# magnification-based gird. At this point, it might be worth reminding yourself how the Bayesian evidence works, by
# going back to the 'introduction' text file.

# So, why do you think why adapting to the source's brightness increases the evidence?

# Its because, by adapting to the source's morphology, we can now access solutions that fit the instrument really well
# (e.g. to the Gaussian noise-limit) but use significantly fewer source-pixels than other grids. For instance, a
# typical magnification based grid uses resolutions of 40 x 40, or 1600 pixels. In contrast, a morphology based grid
# typically uses just 300-800 pixels (depending on the source itself). Clearly, the easiest way to make our source
# solution simpler is to use fewer pixels overall!

# This provides a second benefit. If the best solutions in our fit want to use the fewest source-pixels possible, and
# PyAutoLens can now access those solutions, this means that hyper-mode will run much faster than the
# magnification based grid! Put simply, fewer source-pixels means lower computational overheads. YAY!

# Tutorial 2 done, next up, adaptive regularization!
