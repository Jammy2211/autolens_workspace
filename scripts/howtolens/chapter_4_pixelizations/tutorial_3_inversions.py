"""
Tutorial 3: Inversions
======================

In the previous two tutorials, we introduced:

 - `Pixelization`'s: which place a pixel-grid in the source-plane.
 - `Mappers`'s: which describe how each source-pixel maps to one or more image pixels.

However, non of this has actually helped us fit strong lens data or reconstruct the source galaxy. This is the subject
of this tutorial, where the process of reconstructing the source's light on the pixelization is called an `Inversion`.
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

"""
Lets create an annular mask which traces the stongly lensed source's ring of light.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.5,
    outer_radius=2.8,
)

visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(data=True)

"""
We now create the masked source-plane grid via the tracer, as we did in the previous tutorial.
"""
dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer = al.Tracer(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization.over_sampler.over_sampled_grid
)[1]

"""
we again use the rectangular pixelization to create the mapper.

(Ignore the regularization input below for now, we will cover this in the next tutorial).
"""
mesh = al.mesh.Rectangular(shape=(25, 25))

pixelization = al.Pixelization(mesh=mesh)

mapper_grids = pixelization.mapper_grids_from(
    mask=mask, source_plane_data_grid=source_plane_grid
)
mapper = al.Mapper(
    mapper_grids=mapper_grids,
    over_sampler=dataset.grids.over_sampler_pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

include = aplt.Include2D(mask=True, mapper_source_plane_data_grid=True)

mapper_plotter = aplt.MapperPlotter(mapper=mapper, include_2d=include)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__Pixelization__

Finally, we can now use the `Mapper` to reconstruct the source via an `Inversion`. I'll explain how this works in a 
second, but lets just go ahead and create the inversion first. 
"""
inversion = al.Inversion(dataset=dataset, linear_obj_list=[mapper])

"""
The inversion has reconstructed the source's light on the rectangular pixel grid, which is called the 
`reconstruction`. This source-plane reconstruction can be mapped back to the image-plane to produce the 
`mapped_reconstructed_image`.
"""
print(inversion.reconstruction)
print(inversion.mapped_reconstructed_image)

"""
Both of these can be plotted using an `InversionPlotter`.

It is possible for an inversion to have multiple `Mapper`'s, therefore for certain figures we specify the index 
of the mapper we wish to plot. In this case, because we only have one mapper we specify the index 0.
"""
include = aplt.Include2D(mask=True)

inversion_plotter = aplt.InversionPlotter(inversion=inversion, include_2d=include)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
There we have it, we have successfully reconstructed the source using a rectangular pixel-grid. Whilst this source 
was simple (a blob of light in the centre of the source-plane), inversions come into their own when fitting sources 
with complex morphologies. 

Lets use an inversion to reconstruct a complex source!
"""
dataset_name = "source_complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.figures_2d(data=True)

"""
This code is doing all the same as above (setup the mask, galaxy, tracers, mapper, inversion, etc.).
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.1,
    outer_radius=3.2,
)

visuals = aplt.Visuals2D(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(data=True)

dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grid_2d_list_from(
    grid=dataset.grids.pixelization.over_sampler.over_sampled_grid
)[1]

mapper_grids = mesh.mapper_grids_from(
    mask=mask, source_plane_data_grid=source_plane_grid
)
mapper = al.Mapper(
    mapper_grids=mapper_grids,
    over_sampler=dataset.grids.over_sampler_pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)


inversion = al.Inversion(dataset=dataset, linear_obj_list=[mapper])

"""
Now lets plot the complex source reconstruction.
"""
inversion_plotter = aplt.InversionPlotter(inversion=inversion, include_2d=include)
inversion_plotter.figures_2d(reconstructed_image=True)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
Pretty great, huh? If you ran the complex source pipeline in chapter 3, you'll remember that getting a model image 
that looked this good simply *was not possible*. With an inversion, we can do this with ease and without having to 
perform model-fitting with 20+ parameters for the source's light!

We will now briefly discuss how an inversion actually works, however the explanation I give in this tutorial will be 
overly-simplified. To be good at lens modeling you do not need to understand the details of how an inversion works, you 
simply need to be able to use an inversion to model a strong lens. 

To begin, lets consider some random mappings between our mapper`s source-pixels and the image.
"""
visuals = aplt.Visuals2D(pix_indexes=[[445], [285], [313], [132], [11]])

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper, visuals_2d=visuals, include_2d=include
)
mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
These mappings are known before the inversion reconstructs the source galaxy, which means before this inversion is
performed we know two key pieces of information:

 1) The mappings between every source-pixel and sets of image-pixels.
 2) The flux values in every observed image-pixel, which are the values we want to fit successfully.

It turns out that with these two pieces of information we can linearly solve for the set of source-pixel fluxes that 
best-fit (e.g. maximize the log likelihood) our observed image. Essentially, we set up the mappings between source and 
image pixels as a large matrix and solve for the source-pixel fluxes in an analogous fashion to how you would solve a 
set of simultaneous linear equations. This process is called a `linear inversion`.

There are three more things about a linear inversion that are worth knowing:

 1) When performing fits using light profiles, we discussed how a `model_image` was generated by convolving the light
 profile images with the data's PSF. A similar blurring operation is incorporated into the inversion, such that it 
 reconstructs a source (and therefore image) which fully accounts for the telescope optics and effect of the PSF.

 2) You may be familiar with image sub-gridding, which splits each image-pixel into a sub-pixel (if you are not 
 familiar then feel free to checkout the optional **HowToLens** tutorial on sub-gridding. If a sub-grid is used, it is 
 the mapping between every sub-pixel and source-pixel that is computed and used to perform the inversion. This prevents 
 aliasing effects degrading the image reconstruction. By default **PyAutoLens** uses sub-gridding of degree 4x4.

 3) The inversion`s solution is regularized. But wait, that`s what we'll cover in the next tutorial!

Finally, let me show you how easy it is to fit an image with an `Inversion` using a `FitImaging` object. Instead of 
giving the source galaxy a light profile, we simply pass it a `Pixelization` and regularization, and pass it to a 
tracer.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
Then, like before, we pass the imaging and tracer `FitImaging` object. 

We see some pretty good looking residuals, we must be fitting the lensed source accurately! In fact, we can use the
`subplot_of_planes` method to specifically visualize the inversion and plot the source reconstruction.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

include = aplt.Include2D(mask=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
__Positive Only Solver__

All pixelized source reconstructions use a positive-only solver, meaning that every source-pixel is only allowed
to reconstruct positive flux values. This ensures that the source reconstruction is physical and that we don't
reconstruct negative flux values that don't exist in the real source galaxy (a common systematic solution in lens
analysis).

It may be surprising to hear that this is a feature worth pointing out, but it turns out setting up the linear algebra
to enforce positive reconstructions is difficult to make efficient. A lot of development time went into making this
possible, where a bespoke fast non-negative linear solver was developed to achieve this.

Other methods in the literature often do not use a positive only solver, and therefore suffer from these 
unphysical solutions, which can degrade the results of lens model in general.

__Wrap Up__

And, we're done, here are a few questions to get you thinking about inversions:

 1) The inversion provides the maximum log likelihood solution to the observed image. Is there a problem with seeking 
 the highest likelihood solution? Is there a risk that we're going to fit other things in the image than just the 
 lensed source galaxy? What happens if you reduce the `coefficient` of the regularization object above to zero?

 2) The exterior pixels in the rectangular pixel-grid have no image-pixels in them. However, they are still given a 
 reconstructed flux. Given these pixels do not map to the data, where is this value coming from?
 
__Detailed Explanation__

If you are interested in a more detailed description of how inversions work, then checkout the file
`autolens_workspace/*/imaging/log_likelihood_function/inversion.ipynb` which gives a visual step-by-step
guide of the process alongside equations and references to literature on the subject.
"""
