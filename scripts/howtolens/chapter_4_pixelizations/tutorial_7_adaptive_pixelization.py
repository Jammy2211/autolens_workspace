"""
Tutorial 7: Adaptive Pixelization
=================================

In this tutorial we will introduce a new `Pixelization` object, which uses an `Overlay` image-mesh and a `Delaunay`
mesh.

This pixelization does not use a uniform grid of rectangular pixels, but instead uses a `Delaunay` triangulation.

So, why would we want to do that? Lets take another quick look at the rectangular grid.
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

We'll use the same strong lensing data as the previous tutorial, where:

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

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
The lines of code below do everything we're used to, that is, setup an image, mask it, trace it via a tracer, 
setup the rectangular mapper, etc.
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

dataset = dataset.apply_mask(mask=mask)

pixelization = al.Pixelization(
    mesh=al.mesh.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=0.5),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

include = aplt.Include2D(mask=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
__Advantages and Disadvatanges__

Lets think about the rectangular pixelization. Is this the optimal way to reconstruct our source? Are there features 
in the source-plane that arn`t ideal? How do you think we could do a better job?

There are a number of reasons the rectangular pixelization is not optimal, and is infact a pretty poor method to 
model strong lenses!

So what is wrong with the grid? Well, lets think about the source reconstruction.
"""
include = aplt.Include2D(mapper_source_plane_mesh_grid=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_of_planes(plane_index=1)

"""
There is one clear problem, we are using only a small number of the total source pixels to reconstruct the source. The 
majority of source pixels are located away from the source. By my estimate, we are using just 16 pixels (the central 
4x4 grid) out of the 1600 pixels to actually fit the data! The remaining ~1500 pixels are doing nothing but fitting 
noise. 

This means that regularization is sub-optimal. In tutorial 4, we discussed how the Bayesian evidence of the 
regularization favours the simplest source solution. That is, the solution which fits the data using the fewest source 
pixels. If we are dedicating a large number of source pixels to fitting *nothing*, the source reconstruction is 
unnecessarily complex (and therefore is lower `log_evidence` solution).

If our pixelization could 'focus' its pixels where we actually have more data, e.g. the highly magnified regions of 
the source-plane, we could reconstruct the source using fewer pixels. This would significantly increase the Bayesian
evidence. It'd also be benefitial computationally, as using fewer source pixels means faster run times.

This is what the Delaunay mesh enables.

__Image Mesh__

The Delaunay mesh is an irregular grid of pixels (or triangles) in the source-plane. We must first therefore determine
a set of (y,x) source-plane coordinates defining this grid, specifically where each triangle vertex is loated.

We do this using an `image_mesh`, which defines a method to determine a set of coordinates in the image-plane 
which are ray-traced to the source-plane. These traced coordinates are the triangle vertexes of our source-pixel mesh. 

Below, we use the `Overlay` image-mesh to do this, which overlays a grid of (y,x) coordinates over the image-plane
mask and retains all (Y,x) coordinates which fall within this mask.
"""
image_mesh = al.image_mesh.Overlay(shape=(20, 20))

image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(mask=dataset.mask)

"""
We can plot this grid over the image, to see that it is a coarse grid of (y,x) coordinates laid ove the image.
"""
visuals = aplt.Visuals2D(grid=image_plane_mesh_grid, mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(data=True)

"""
By passing a `Tracer` a source galaxy with the image-mesh and a `Delaunay` mesh object, contained in 
a `Pixelization` object, it automatically computes this source-plane Delaunay mesh.
"""
pixelization = al.Pixelization(
    image_mesh=image_mesh,
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
By using this tracer in a fit, we see that our source-plane no longer uses rectangular pixels, but a Delaunay mesh!
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

include = aplt.Include2D(
    mask=True, mapper_image_plane_mesh_grid=True, mapper_source_plane_mesh_grid=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
A closer inspection of the pixelization shows the improvement. 

We are using fewer pixels than the rectangular grid (400, instead of 1600) and reconstructing the source is far 
greater detail!
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_of_planes(plane_index=1)

"""
__Regularization__

On the rectangular grid, we regularized each source pixel with its 4 neighbors. We compared their fluxes, summed 
the differences, and penalized solutions where the differences were large. 

For a Delaunay grid, we do a similar calculation, instead comparing each source-pixel with the 3 Delaunay triangles 
it shares a direct vertex with.

__Wrap Up__

The `Overlay` image-mesh and `Delaunay` mesh is still far from optimal. There are lots of source-pixels effectively f
itting just noise. We can achieve even better solutions if the central regions of the source were reconstructed using 
more pixels and fewer source pixels are used in the outskirts of the source plane. 

Tutorials 9, 10 and 11 show even more advanced and adaptive pixelizations which do just this, by adapting to the
source galaxy's morphology rather than the mass model magnification.

In the mean time, you may wish to experiment with using both Rectangular and Delaunay grids to fit 
lenses which can be easily achieve by changing the input pixelization given to a pipeline.
"""
