"""
Tutorial 10: Brightness Adaption
================================

In the previous tutorial we motivated our need to adapt the pixelization to the source's morphology, such that source
pixels congregates in the source's brightest regions regardless of where the source is located in the source-plane. 

This poses a challenge; how do we adapt our pixelization to the reconstructed source's light, before we've
actually reconstructed the source and therefore know what to adapt it too?

To do this, we define 'adapt_images' of the lensed source galaxy, which are model images of the source computed using 
a previous lens model that has been fit to the image (e.g. in the earlier search of a pipeline). This image tells
us where in the image our source is located, thus informing us of where we need to adapt our source pixelization!

This tutorial goes into the details of how this works. We'll use the same compact source galaxy as the previous
tutorial and begin by fitting it with a magnification based pixelization. This will produce a model image which can
then be used an adapt image.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

we'll use the same strong lensing data as the previous tutorial, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=3.0,
)

dataset = dataset.apply_mask(mask=mask)

"""
Next, we're going to fit the image using the Delaunay magnification based grid. 

The code below does all the usual steps required to do this.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=3.3),
)

source_galaxy_magnification = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy_magnification])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
Lets have a quick look to make sure it has the same residuals we saw in tutorial 1.
"""
include = aplt.Include2D(
    mask=True, mapper_image_plane_mesh_grid=True, mapper_source_plane_mesh_grid=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
__Adapt Image__

We can use this fit to set up our adapt image. 

This adapt-image is not perfect, because there are residuals in the central regions of the reconstructed source. 
However, it is good enough for us to adapt our pixelization to the lensed source.
"""
adapt_image = fit.model_data.slim

"""
__Adaption__

Now lets take a look at brightness based adaption in action. 

Below, we define a source-galaxy using the `Hilbert` image-mesh (we discuss below how this adapts to the source light) 
and `Delaunay` mesh and use this to fit the lens-data. 
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=10.0),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=0.5),
)

galaxy_adapt = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
)

"""
The adapt image is paired to the galaxy that it represents and that it is used to adapt the mesh too.

This uses the `AdaptImages` object, which receives a dictionary mapping every galaxy to its adapt image.
"""
adapt_images = al.AdaptImages(galaxy_image_dict={galaxy_adapt: adapt_image})

"""
We now fit using this adapt image and mesh using the normal API.

Note however that the `FitImaging` object receives the `adapt_images` as an input and they are used when
setting up the image-mesh and mesh.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, galaxy_adapt])

fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)


"""
Our reconstruction of the image no longer has residuals! 

By congregating more source pixels in the brightest regions of the source reconstruction we get a better fit. 
Furthermore, we can check that this provides an increase in Bayesian log evidence, noting that the log evidence of the 
compact source when using a `Overlay` image-mesh was 4216:
"""
print("Evidence using magnification based pixelization. ", 4216)
print("Evidence using brightness based pixelization. ", fit.log_evidence)

"""
It increases by over 1000, which for a Bayesian evidence is pretty large! 

This pixelization is a huge success, we should have been adapting to the source's brightness all along! 

In doing so, we will *always* reconstruct the detailed structure of the source's brightest regions with a 
sufficiently high resolution!

We are now able to adapt the pixelization's mesh to the morphology of the lensed source galaxy. To my knowledge, this
is the *best* approach one can take in lens modeling. Its more tricky to implement and introduces additional non-linear 
parameters. But the pay-off is worth it, as we fit our data better and use fewer source pixels to reconstruct
the source.

__Hilbert__

So how does the `adapt_image` adapt the pixelization to the source's brightness? It uses a standard algorithm for 
partitioning data in statistics called a Hilbert curve:

https://en.wikipedia.org/wiki/Hilbert_curve

In simple terms, this algorithm works as follows:

 1) Input an image of weight values to the Hilbert algorithm which determines the "hilbert space filling curve" 
 (e.g. this `weight_map` is determined from the adapt-image). The Hilbert space-filling curve fills more space there
 the weight values are highest, and less space where they are lowest. It therefore closely traces the brightest
 regions of the image.
    
 2) Probabilistically draw N $(y,x)$ points from this Hilbert curve, where the majority of points will therefore be 
 drawn from high weighted regions. 
    
 3) These N $(y,x)$ are our source-pixel centres, albeit we have drawn them in the image-plane so we first map them
 to the source-plane, via the mass model, in order to set up the source pixel centres of the mesh. Because points are
 drawn from high weighted regions (e.g. the brightest image pixels in the lensed source adapt image), we will trace 
 more source-pixels to the brightest regions of where the source is actually reconstructed.
 
__Weight Map__

We now have a sense of how our `Hilbert` image-mesh is computed, so lets look at how we create the weighted data the 
Hilbert space filling curve uses.

This image, called the `weight_map` is generated using the `weight_floor` and `weight_power` parameters of 
the `Hilbert` object. The weight map is generated following 4 steps:

 1) Take an absolute value of all pixels in the adapt image, because negative values break the Hilbert algorithm.
    
 2) Divide all values of this image by its maximum value, such that the adapt-image now only contains values between 
 0.0 and 1.0 (where the values of 1.0 are the maximum values of the adapt-image).
    
 3) Add the weight_floor to all values (a weight_floor of 0.0 therefore does not change the weight map).
    
 4) Raise all values to the power of weight_power (a weight_power of 1.0 therefore does not change the
 weight map, whereas a value of 0.0 means all values 1.0 and therefore weighted equally).
 
The idea is that using high values of `weight_power` will make the highest weight values much higher than the lowest
values, such that the Hilbert curve will trace these values much more than the lower values. The weight_floor gives
the algorithm some balance, by introducing a floor to the weight map that prevents the lowest values from being
weighted to near zero.

Lets look at this in action. we'll inspect 3 weight_maps, using a weight_power of 0.0, 5.0 and 10.0 and
setting the `weight_floor` to 0.0 for now.
"""
image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=0.0)

image_weight_power_0 = image_mesh.weight_map_from(adapt_data=adapt_image)
image_weight_power_0 = al.Array2D(values=image_weight_power_0, mask=mask)

array_plotter = aplt.Array2DPlotter(
    array=image_weight_power_0, visuals_2d=aplt.Visuals2D(mask=mask)
)
array_plotter.figure_2d()


image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=5.0)

image_weight_power_5 = image_mesh.weight_map_from(adapt_data=adapt_image)
image_weight_power_5 = al.Array2D(values=image_weight_power_5, mask=mask)

array_plotter = aplt.Array2DPlotter(
    array=image_weight_power_0, visuals_2d=aplt.Visuals2D(mask=mask)
)
array_plotter.figure_2d()


image_mesh = al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=10.0)

image_weight_power_10 = image_mesh.weight_map_from(adapt_data=adapt_image)
image_weight_power_10 = al.Array2D(values=image_weight_power_10, mask=mask)

array_plotter = aplt.Array2DPlotter(
    array=image_weight_power_0, visuals_2d=aplt.Visuals2D(mask=mask)
)
array_plotter.figure_2d()

"""
When we increase the weight-power the brightest regions of the adapt-image become weighted higher relative to the 
fainter regions. 

This means that the Hilbert algorithm will adapt more pixels to the brightest regions of the source.

Lets use the method to perform a fit with a weight power of 10, showing that we now get a significantly higher
log_evidence.
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Hilbert(pixels=500, weight_floor=0.0, weight_power=10.0),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_weight_power_10 = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_weight_power_10])

adapt_images = al.AdaptImages(galaxy_image_dict={source_weight_power_10: adapt_image})

fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
So, what does the `weight_floor` do? Increasing the weight-power congregates pixels around the source. However, there 
is a risk that by congregating too many source pixels in its brightest regions we lose resolution further out, where 
the source is bright, but not its brightest!

The `weight_floor` allows these regions to maintain a higher weighting whilst the `weight_power` increases. This means 
that the mesh can fully adapt to the source's brightest and faintest regions simultaneously.

Lets look at once example:
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Hilbert(pixels=500, weight_floor=0.5, weight_power=10.0),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_weight_floor = al.Galaxy(
    redshift=1.0, pixelization=pixelization, adapt_galaxy_image=adapt_image
)

weight_floor = source_weight_floor.pixelization.image_mesh.weight_map_from(
    adapt_data=adapt_image
)
weight_floor = al.Array2D(values=weight_floor, mask=mask)

array_plotter = aplt.Array2DPlotter(
    array=image_weight_power_0, visuals_2d=aplt.Visuals2D(mask=mask)
)
array_plotter.figure_2d()

tracer = al.Tracer(galaxies=[lens_galaxy, source_weight_floor])

adapt_images = al.AdaptImages(galaxy_image_dict={source_weight_floor: adapt_image})

fit = al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
__Wrap Up__

To end, lets think about the Bayesian evidence, which we saw now goes to significantly higher values than for a 
magnification-based grid. At this point, it might be worth reminding yourself how the Bayesian evidence works by 
going back to description in this chapters `introduction` text file.

So, why do you think why adapting to the source's brightness increases the log evidence?

It is because by adapting to the source's morphology we can now access solutions that fit the data really well 
(e.g. to the Gaussian noise-limit) but use significantly fewer source-pixels than before. For instance, a typical 
magnification based grid uses resolutions of 40 x 40, or 1600 pixels. In contrast, a morphology based pixelization 
typically uses just 300-800 pixels (depending on the source itself). Clearly, the easiest way to make our source 
solution simpler is to use fewer pixels overall!

This provides a second benefit. If the best solutions in our fit want to use the fewest source-pixels possible and 
**PyAutoLens** can now access those solutions, this means that adapt-mode will run much faster than the magnification 
based grid! Put simply, fewer source-pixels means lower computational overheads. YAY!

Tutorial 2 done, next up, adaptive regularization!
"""
