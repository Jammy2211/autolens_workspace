import autolens as al

# In the previous tutorial we motivated a need to adapt our pixelization to the source's morphology, such that source
# pixels congregates in the source's brightest regions regardless of where it is located in the source-plane. This
# raises an interesting question; how do we adapt our source pixelization to the reconstructed source, before we've
# actually reconstructed the source and therefore know what to adapt it too?

# To do this, we define a 'hyper-galaxy-image' of the lensed source galaxy. This is a model image of the source computed
# using a previous lens model fit to the image (e.g. in an earlier phase of a pipeline). This image tells us where in
# the image our source is located, thus telling us where we need to adapt our source pixelization!

# So, lets go into the details of how this works. We'll use the same compact source galaxy as the previous tutorial and
# we'll begin by fitting it with a magnification based pixelization. Why? So we can use its model image to set up
# the hyper-galaxy-image.

# This is the usual simulate function, using the compact source of the previous tutorial.


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
        noise_seed=1,
    )


# Lets simulate the data, draw a 3.0" mask and set up the lens data that we'll fit.

ccd_data = simulate()
mask = al.Mask.circular(shape=(150, 150), pixel_scale=0.05, radius_arcsec=3.0)

# To perform brightness adaption, we use a 'binned grid' in our lens_data aongide binned_hyper_galaxy_images. For now,
# ignore these 'binned' quantities - I'll explain what they do at the end of the tutorial.
lens_data = al.LensData(
    ccd_data=ccd_data, mask=mask, pixel_scale_binned_grid=ccd_data.pixel_scale
)

# Next, we're going to fit the image using our magnification based grid. The code below does all the usual steps
# required to do this.

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
    ),
)

source_galaxy_magnification = al.Galaxy(
    redshift=1.0,
    pixelization=al.pixelizations.VoronoiMagnification(shape=(30, 30)),
    regularization=al.regularization.Constant(coefficient=3.3),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy_magnification])

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

# Lets have a quick look to make sure it has the same residuals we saw in tutorial 1.

al.lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

al.inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# Finally, we can use this fit to set up our hyper-galaxy-image. This hyper-galaxy-image isn't perfect,  as there are
# residuals in the central regions of the reconstructed source. But it's *okay* and it'll certainly give us enough
# information on where the lensed source is located to adapt our pixelization.
hyper_image_1d = fit.model_image(return_in_2d=False)

# Now lets take a look at brightness based adaption in action! Below, we define a source-galaxy using our new
# 'VoronoiBrightnessImage' pixelization and use this to fit the lens-data. One should note that we also attach the
# hyper-galaxy-image to this galaxy because its pixelization uses this hyper-galaxy-image for adaption, thus the
# galaxy needs to know what hyper-galaxy-image it should use!

source_galaxy_brightness = al.Galaxy(
    redshift=1.0,
    pixelization=al.pixelizations.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=10.0
    ),
    regularization=al.regularization.Constant(coefficient=0.5),
    hyper_galaxy_image_1d=hyper_image_1d,
    binned_hyper_galaxy_image_1d=hyper_image_1d,
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy_brightness])

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

al.lens_fit_plotters.plot_fit_subplot(
    fit=fit,
    should_plot_image_plane_pix=True,
    should_plot_mask=True,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

al.inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# Would you look at that! Our reconstruction of the image no longer has residuals! By congregating more source pixels
# in the brightest regions of the source reconstruction we get a better fit. Furthermore, we can check that this
# provides an increase in Bayesian evidence, noting that the evidence of the compact source when using a
# VoronoiMagnification pixelization was 14236:

print("Evidence using magnification based pixelization = ", 14236.292117135737)

print("Evidence using brightness based pixelization = ", fit.evidence)


# It increases! By over 200, which, for a Bayesian evidence, is pretty damn large! By any measure, this pixelization is
# a huge success. It turns out that we should have been adapting to the source's brightness all along! In doing so,
# we will *always* reconstruct the detailed structure of the source's brightest regions with a sufficiently
# high resolution. Hurrah!

# So, we are now able to adapt our pixelization to the morphology of our lensed source galaxy. To my knowledge,
# this is the *best* approach one can take in lens modeling. Its more tricky to implement (as I'll explain next)
# and introduces extra hyper-galaxy-parametersr. But the pay-off is more than worth it, as we fit our image data
# better and (typically) end up using far fewer source pixels to fit the data because we don't 'waste' pixels
# reconstructing regions of the source-plane where there is no signal.


# Okay, so how does the hyper_image actually adapt our pixelization to the source's brightness? It uses a 'weighted
# KMeans clustering algorithm', which is a standard algorithm for partioning data in statistics.

# In simple terms, this algorithm works as follows:

# 1) Give the KMeans algorithm a set of weighted data (e.g. determined from the hyper-galaxy image).

# 2) For a given number of K-clusters, this algorithm will find a set of (y,x) coordinates that equally partition
#    the weighted data-set. Wherever the data has higher weighting, more clusters congregate and visa versa.

# 3) The returned (y,x) 'clusters' then make up our source-pixel centres, where the brightest (e.g. higher weighted)
#    regions of the hyper-galaxy-image will have more clusters! Like we did for the magnification based pixelization,
#    we can then trace these coordinates to the source-plane to define our source-pixel pixelization.

# This is a fairly simplistic description of a KMeans algorithm. Feel free to check out the links below for a more
# in-depth view:

# https://en.wikipedia.org/wiki/K-means_clustering
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html


# Okay, so we now have a sense of how our VoronoiBrightnessImage pixelization is computed. Now, lets look at how we
# create the weighted data the KMeans algorithm uses.

# This image, called the 'cluster_weight_map' is generated using the 'weight_floor' and 'weight_power' parameters
# of the VoronoiBrightness pixelization. The cluster weight map is generated following 4 steps:

# 1) Increase all values of the hyper-galaxy-image that are < 0.02 to 0.02. This is necessary because negative values
#    and zeros break the KMeans clustering algorithm.

# 2) Divide all values of this image by its maximum value, such that the hyper-galaxy-image now only contains values between
#    0.0 and 1.0 (where the values of 1.0 were the maximum values of the hyper-galaxy-image).

# 3) Add the weight_floor to all values (a weight_floor of 0.0 therefore does not change the cluster weight map).

# 4) Raise all values to the power of weight_power (a weight_power of 1.0 therefore does not change the cluster weight
#    map, whereas a value of 0.0 means all values 1.0 and therefore weighted equally).

# Lets look at this in action. We'll inspect 3 cluster_weight_maps, using a weight_power of 0.0, 5.0 and 10.0,
# setting the weight_floor to 0.0 such that it does not change the cluster weight map.

source_weight_power_0 = al.Galaxy(
    redshift=1.0,
    pixelization=al.pixelizations.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=0.0
    ),
    regularization=al.regularization.Constant(coefficient=1.0),
    hyper_galaxy_image_1d=hyper_image_1d,
    binned_hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_power_0 = source_weight_power_0.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_power_0.hyper_galaxy_image_1d
)
cluster_weight_power_0 = mask.scaled_array_2d_from_array_1d(
    array_1d=cluster_weight_power_0
)
al.array_plotters.plot_array(
    array=cluster_weight_power_0,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

source_weight_power_5 = al.Galaxy(
    redshift=1.0,
    pixelization=al.pixelizations.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=5.0
    ),
    regularization=al.regularization.Constant(coefficient=1.0),
    hyper_galaxy_image_1d=hyper_image_1d,
    binned_hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_power_5 = source_weight_power_5.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_power_5.hyper_galaxy_image_1d
)
cluster_weight_power_5 = mask.scaled_array_2d_from_array_1d(
    array_1d=cluster_weight_power_5
)
al.array_plotters.plot_array(
    array=cluster_weight_power_5,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

source_weight_power_10 = al.Galaxy(
    redshift=1.0,
    pixelization=al.pixelizations.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.0, weight_power=10.0
    ),
    regularization=al.regularization.Constant(coefficient=1.0),
    hyper_galaxy_image_1d=hyper_image_1d,
    binned_hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_power_10 = source_weight_power_10.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_power_10.hyper_galaxy_image_1d
)
cluster_weight_power_10 = mask.scaled_array_2d_from_array_1d(
    array_1d=cluster_weight_power_10
)
al.array_plotters.plot_array(
    array=cluster_weight_power_10,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

# When we increase the weight-power the brightest regions of the hyper-galaxy-image become weighted higher relative to
# the fainter regions. This means that t e KMeans algorithm will adapt its pixelization to the brightest regions of
# the source.


tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_weight_power_0])

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

al.inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_weight_power_5])

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

al.inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_weight_power_10])

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

al.inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# So, what does the weight_floor do? Increasing the weight-power congregates pixels around the source. However,
# there is a risk that by congregating too many source pixels in its brightest regions we lose resolution further out,
# where the source is bright, but not its brightest!

# The noise-floor allows these regions to maintain a higher weighting whilst the noise_power increases. This means
# that the pixelization can fully adapt to the source's brightest and faintest regions simultaneously.

# Lets look at once example:

source_weight_floor = al.Galaxy(
    redshift=1.0,
    pixelization=al.pixelizations.VoronoiBrightnessImage(
        pixels=500, weight_floor=0.5, weight_power=10.0
    ),
    regularization=al.regularization.Constant(coefficient=1.0),
    hyper_galaxy_image_1d=hyper_image_1d,
    binned_hyper_galaxy_image_1d=hyper_image_1d,
)

cluster_weight_floor = source_weight_floor.pixelization.cluster_weight_map_from_hyper_image(
    hyper_image=source_weight_floor.hyper_galaxy_image_1d
)
cluster_weight_floor = mask.scaled_array_2d_from_array_1d(array_1d=cluster_weight_floor)

al.array_plotters.plot_array(
    array=cluster_weight_floor,
    mask=mask,
    extract_array_from_mask=True,
    zoom_around_mask=True,
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_weight_floor])

fit = al.LensDataFit.for_data_and_tracer(lens_data=lens_data, tracer=tracer)

al.inversion_plotters.plot_pixelization_values(
    inversion=fit.inversion, should_plot_centres=True
)

# Now, lets addresses why the functions above used a binned-grid and binned hyper images. Basically, running a weighted
# KMeans algorithm is comoputationally expensive. Furthermore, the higher resolutiion the data you run it on, the
# longer it takes to run. Too work around this, we used a binned hyper image, which rebins the hyper-data to a sparse
# resolution. We can still use this image to adapt the source pixelization fine, it simply means the computational run
# times are reduced.


# To end, lets think about the Bayesian evidence which goes to significantly higher values than a magnification-based
# gird. At this point, it might be worth reminding yourself how the Bayesian evidence works by going back to the
# 'introduction' text file.

# So, why do you think why adapting to the source's brightness increases the evidence?

# It is because by adapting to the source's morphology we can now access solutions that fit the data really well
# (e.g. to the Gaussian noise-limit) but use significantly fewer source-pixels than other grids. For instance, a
# typical magnification based grid uses resolutions of 40 x 40, or 1600 pixels. In contrast, a morphology based grid
# typically uses just 300-800 pixels (depending on the source itself). Clearly, the easiest way to make our source
# solution simpler is to use fewer pixels overall!

# This provides a second benefit. If the best solutions in our fit want to use the fewest source-pixels possible and
# PyAutoLens can now access those solutions, this means that hyper-galaxy-mode will run much faster than the
# magnification based grid! Put simply, fewer source-pixels means lower computational overheads. YAY!

# Tutorial 2 done, next up, adaptive regularization!
