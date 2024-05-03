"""
Tutorial 6: Lens Modeling
=========================

When modeling complex source's with parametric profiles, we quickly entered a regime where our non-linear search was
faced with a parameter space of dimensionality N=20+ parameters. This made the model-fitting inefficient and likely to
infer a local maxima.

Inversions do not suffer this problem, meaning they are a very a powerful tool for modeling strong lenses. Furthermore,
they have *more* freemdom than parametric light profiles because they do not relying on specific analytic light
distributions and a symmetric profile shape. This will allow us to fit more complex mass models and ask ever more
interesting scientific questions!

However, inversion do have some short comings that we need to be aware of before we use them for lens modeling. That`s
what we cover in this tutorial.
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
    radius=2.5,
)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
This function fits the imaging data with a tracer, returning a `FitImaging` object.
"""


def perform_fit_with_lens__source_galaxy(dataset, lens_galaxy, source_galaxy):
    mask = al.Mask2D.circular_annular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        inner_radius=0.5,
        outer_radius=2.2,
    )

    dataset = dataset.apply_mask(mask=mask)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(dataset=dataset, tracer=tracer)


"""
__Unphysical Solutions__

The code below illustrates a systematic set of solutions called demagnified solutions, which negatively impact
lens modeling using source pixelizations.

Since writing the code below, I have wrote a full readthedocs page illustrating the issue, which is linked too below.
I recommend you read this page first, to understand what a demagnified solution is, why its a problem and how we
fix it. The code below should then build on this.

 https://pyautolens.readthedocs.io/en/latest/general/demagnified_solutions.html

__Brief Description__

To see the short-comings of an inversion, we begin by performing a fit where the lens galaxy has an incorrect 
mass-model (I've reduced its Einstein Radius from 1.6 to 0.8). This is a mass model the non-linear search may sample at 
the beginning of a model-fit.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=0.8,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

pixelization = al.Pixelization(
    mesh=al.mesh.Rectangular(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

fit = perform_fit_with_lens__source_galaxy(
    dataset=dataset, lens_galaxy=lens_galaxy, source_galaxy=source_galaxy
)

include = aplt.Include2D(mask=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
What happened!? This incorrect mass-model provides a really good fit to the image! The residuals and chi-squared-map 
are as good as the ones we saw in the previous tutorials.

How can an incorrect lens model provide such a fit? Well, as I'm sure you noticed, the source has been reconstructed 
as a demagnified version of the image. Clearly, this is not a physical solution or a solution that we want our 
non-linear search to find, but for inversion's the reality is these solutions eixst.

This is not necessarily problematic for lens modeling. Afterall, the source reconstruction above is extremely complex, 
it requires a lot of source pixels to fit the image accurately and its lack of smoothness will be heavily penalized
by regularization when we compute the Bayesian evidence. Indeed, its Bayesian evidence is much lower than the true lens
model solution:
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
    mesh=al.mesh.Rectangular(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

correct_fit = perform_fit_with_lens__source_galaxy(
    dataset=dataset, lens_galaxy=lens_galaxy, source_galaxy=source_galaxy
)

fit_plotter = aplt.FitImagingPlotter(fit=correct_fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

print("Bayesian Evidence of Incorrect Fit:")
print(fit.log_evidence)
print("Bayesian Evidence of Correct Fit:")
print(correct_fit.log_evidence)

"""
The `log_evidence` *is* lower. However, the difference in `log_evidence` is not *that large*. This could be a problem 
for the non-linear search, as it will see many solutions in parameter space with high `log_evidence` values. Furthermore, 
these solutions occupy a *large volumne* of parameter space (e.g. everywhere the lens model that is wrong). This makes 
it easy for the non-linear search to get lost searching through these unphysical solutions and, unfortunately, inferring 
an incorrect lens model (e.g. a local maxima).

There is no simple fix for this, and it is the price we pay for making the inversion has so much flexibility in how it
reconstructs the source's light. The solution to this problem? Search chaining. In fact, this is the problem that lead
us to initially conceive of search chaining! 

The idea is simple, we write a pipeline that begins by modeling the source galaxy's light using a light profile, thereby
initializing the priors for the lens galaxy's light and mass. Then, when we switch to an `Inversion` in the next 
search, the mass model starts in the correct regions of parameter space and does not get lost sampling these 
incorrect solutions.

The following paper discusses these solutions in more detail (https://arxiv.org/abs/2012.04665).

__Light Profiles__

We can also model strong lenses using light profiles and an inversion at the same time. We do this when we want to 
simultaneously fit and subtract the lens galaxy's light using a light profile whilst reconstructing the source's
light using an inversion. 

To do this, all we have to do is give the lens galaxy a light profile and use the tracer and fit objects we are used 
too:.
"""
dataset_name = "lens_sersic"
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
    radius=2.5,
)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_of_planes(plane_index=1)

"""
When fitting such an image we now want to include the lens's light in the analysis. Lets update our mask to be 
circular so that it includes the central regions of the image and lens galaxy.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=2.5,
)

dataset = dataset.apply_mask(mask=mask)

"""
As I said above, performing this fit is the same as usual, we just give the lens galaxy a `LightProfile`.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.1, 0.0), einstein_radius=1.6),
)

pixelization = al.Pixelization(
    mesh=al.mesh.Rectangular(shape=(20, 20)),
    regularization=al.reg.Constant(coefficient=1.0),
)

source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
This fit will now subtract the lens galaxy's light from the image and fits the resulting source-only image with the 
inversion. When we plot the image, a new panel on the sub-plot appears showing the model image of the lens galaxy.
"""
fit = al.FitImaging(dataset=dataset, tracer=tracer)

include = aplt.Include2D(mask=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
Of course if the lens subtraction is rubbish so is our fit. We can therefore be sure that our lens model will want to 
fit the lens galaxy's light accurately (below, I've decreased the lens galaxy intensity from 1.0 to 0.5 to show the
result of a poor lens light subtraction).
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=0.5,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.1, 0.0), einstein_radius=1.6),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

include = aplt.Include2D(mask=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
__Wrap Up__

And with that, we're done. I'll end by pointing out a few things about what we've covered to get you thinking about 
the next tutorial on adaption.
    
 - When the lens galaxy's light is subtracted perfectly it leaves no residuals. However, if it isn't subtracted 
 perfectly it does leave residuals, which will be fitted by the inversion. If the residual are significant this is 
 going to impact the source reconstruction negatively and can lead to some pretty nasty systematics. In the next 
 chapter, we'll learn how our adaptive analysis can prevent this residual fitting.
"""
