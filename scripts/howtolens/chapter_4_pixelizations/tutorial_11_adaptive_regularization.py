"""
Tutorial 11: Adaptive Regularization
====================================

In tutorial 7, we discussed why the `Constant` regularization scheme was sub-optimal. Different regions of the source
demand different levels of regularization, motivating a regularization scheme which adapts to the reconstructed
source's surface brightness.

This raises the same question as before, how do we adapt our regularization scheme to the source before we've
reconstructed it? Just like in the last tutorial, we'll use a model image of a strongly lensed source from a previous
model that we've begun calling the `adapt-image`.
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
__Convenience Function__

We are going to fit the image using a magnification based grid. 

To perform the fits, we'll use a convenience function to fit the lens data we simulated above.
"""


def fit_via_source_galaxy_from(dataset, source_galaxy, adapt_images=None):
    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0),
            einstein_radius=1.6,
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        ),
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(dataset=dataset, tracer=tracer, adapt_images=adapt_images)


"""
Use the magnification based source to fit this data.
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.Constant(coefficient=3.3),
)

source_magnification = al.Galaxy(redshift=1.0, pixelization=pixelization)

fit = fit_via_source_galaxy_from(dataset=dataset, source_galaxy=source_magnification)

include = aplt.Include2D(
    mask=True, mapper_image_plane_mesh_grid=True, mapper_source_plane_mesh_grid=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
The fit looks just like it did in the previous tutorials (residuals in the centre due to a lack of source pixels). 

Lets quickly remind ourselves that the effective regularization weight of each source pixel is our input coefficient 
value of 3.3.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, regularization_weights=True
)

"""
__Adaptive Regularization__

Lets now look at adaptive regularization in action, by setting up a adapt-image and using the `AdaptiveBrightness` 
regularization scheme. 

This introduces additional parameters, that are explained below.
"""
adapt_image = fit.model_data.slim

pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.005, outer_coefficient=1.9, signal_scale=3.0
    ),
)

source_adaptive_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    adapt_galaxy_image=adapt_image,
)

adapt_images = al.AdaptImages(
    galaxy_image_dict={source_adaptive_regularization: adapt_image}
)

fit = fit_via_source_galaxy_from(
    dataset=dataset,
    source_galaxy=source_adaptive_regularization,
    adapt_images=adapt_images,
)

inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)

"""
As expected, we now have a variable regularization scheme. 

The regularization of the source's brightest regions is much lower than that of its outer regions. As discussed 
before, this is what we want. Lets quickly check that this does, indeed, increase the Bayesian log evidence:
"""
print("Evidence using constant regularization. ", 4216)
print("Evidence using adaptive regularization. ", fit.log_evidence)

"""
Yes, it does! 

Combining the adaptive mesh and regularization will only further benefit lens modeling!

However, as shown below, we don't fit the source as well as the morphology based mesh did in the last chapter. 
This is because although the adaptive regularization scheme improves the fit, the magnification based 
mesh simply does not have sufficient resolution to resolve the source's cuspy central light.
"""
fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()

"""
__How does adaptive regularization work?__

For every source-pixel, we have a mapping between that pixel and a set of pixels in the adapt-image. Therefore, for 
every source-pixel, if we sum the values of all adapt-image pixels that map to it we get an estimate of how much of 
the lensed source's signal we expect will be reconstructed. We call this each pixel's `pixel signal`.

If a source-pixel has a higher pixel-signal, we anticipate that it`ll reconstruct more flux and we use this information 
to regularize it less. Conversely, if the pixel-signal is close to zero, the source pixel will reconstruct near-zero 
flux and regularization will smooth over these pixels by using a high regularization coefficient.

This works as follows:

 1) For every source pixel, compute its pixel-signal, the summed flux of all corresponding image-pixels in the 
 adapt-galaxy-image.
    
 2) Divide every pixel-signal by the number of image-pixels that map directly to that source-pixel. In doing so, all 
 pixel-signals are 'relative'. This means that source-pixels which by chance map to more image-pixels than their 
 neighbors will not have a higher pixel-signal, and visa versa. This ensures the specific pixelization
 does impact the adaptive regularization pattern.
    
 3) Divide the pixel-signals by the maximum pixel signal so that they range between 0.0 and 1.0.
    
 4) Raise these values to the power of the adapt-parameter `signal_scale`. For a `signal_scale` of 0.0, all 
 pixels will therefore have the same final pixel-scale. As the `signal_scale` increases, a sharper transition of 
 pixel-signal values arises between regions with high and low pixel-signals.
    
 5) Compute every source pixel's effective regularization coefficient as:
    
 (inner_coefficient * pixel_signals + outer_coefficient * (1.0 - pixel_signals)) ** 2.0
    
 This uses two regularization coefficient parameters, one which is applied to pixels with high pixel-signals and one to 
 pixels with low pixel-signals. Thus, pixels in the inner regions of the source may be given a lower level of 
 regularization than pixels further away, as desired.

Thus, we now adapt our regularization scheme to the source's surface brightness. Where its brighter (and therefore 
has a steeper flux gradient) we apply a lower level of regularization than further out. Furthermore, in the edges of 
the source-plane where no source-flux is present we will assume a high regularization coefficient that smooths over 
all of the source-pixels.

Try looking at a couple of extra solutions which use with different inner and outer regularization coefficients or 
signal scales. I doubt you'll notice a lot change visually, but the log evidence certainly has a lot of room for 
manuveur with different values.

You may find solutions that raise an `InversionException`. These solutions mean that the matrix used during the 
linear algebra calculation was ill-defined, and could not be inverted. These solutions are removed 
during lens modeling.
"""
pixelization = al.Pixelization(
    image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
    mesh=al.mesh.Delaunay(),
    regularization=al.reg.AdaptiveBrightness(
        inner_coefficient=0.001, outer_coefficient=0.2, signal_scale=2.0
    ),
)

source_adaptive_regularization = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    adapt_galaxy_image=adapt_image,
)

adapt_images = al.AdaptImages(
    galaxy_image_dict={source_adaptive_regularization: adapt_image}
)

fit = fit_via_source_galaxy_from(
    dataset=dataset,
    source_galaxy=source_adaptive_regularization,
    adapt_images=adapt_images,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()

inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, reconstruction=True, regularization_weights=True
)

print("Evidence using adaptive regularization. ", fit.log_evidence)

"""
__Wrap Up__

To end, lets consider what this adaptive regularization scheme means in the context of maximizing the Bayesian
evidence. In the previous tutorial, we noted that by using a brightness-based adaptive pixelization we increased 
the Bayesian evidence by allowing for new solutions which fit the data user fewer source pixels; the key criteria 
in making a source reconstruction 'more simple' and 'less complex'.

As you might of guessed, adaptive regularization increases the Bayesian log evidence by making the source 
reconstruction simpler:

 1) Reducing regularization in the source's brightest regions produces a `simpler` solution in that we are not 
 over-smoothing our reconstruction of its brightest regions.
    
 2) Increasing regularization in the outskirts produces a simpler solution by correlating more source-pixels, 
 effectively reducing the number of pixels used by the reconstruction.

Together, brightness based pixelization's and regularization allow us to find the objectively `simplest` source 
solution possible and therefore ensure that our Bayesian evidence has a well defined maximum value. This was not the 
case for magnification based pixelization's and constant regularization schemes.
"""
