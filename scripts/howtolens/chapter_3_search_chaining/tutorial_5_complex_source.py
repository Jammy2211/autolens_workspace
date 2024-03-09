"""
Tutorial 5: Complex Source
==========================

Up to now, we've not paid much attention to the source galaxy's morphology. We've assumed its a single-component
exponential profile, which is a fairly crude assumption. A quick look at any image of a real galaxy reveals a
wealth of different structures that could be present: bulges, disks, bars, star-forming knots and so on. Furthermore,
there could be more than one source-galaxy!

In this example, we'll explore how far we get fitting a complex source using a pipeline. Fitting complex source's is
an exercise in diminishing returns. Each light profile we add to our source model brings with it an extra 5-7,
parameters. If there are 4 components, or multiple galaxies, we are quickly entering the somewhat nasty regime of
30-40+ parameters in our non-linear search. Even with a pipeline, that is a lot of parameters to fit!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

we'll use new strong lensing data, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is four linear `Sersic`.
"""
dataset_name = "source_complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.05,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Paths__

All four searches will use the same `path_prefix`, so we write it here to avoid repetition.
"""
path_prefix = path.join("howtolens", "chapter_3", "tutorial_4_complex_source")

"""
__Search Chaining Approach__

The source is clearly complex, with more than 4 peaks of light. Through visual inspection of this image, we cannot state
with confidence how many sources of light there truly is! The data also omits he lens galaxy's light. This keep the 
number of parameters down and therefore makes the searches faster, however we would not get such a luxury for a real 
galaxy.

To fit this lens with a complex source model, our approach is simply to fit the lens galaxy mass and source using
one light profile in the first search, and then add an additional light profile to each search. The mass model and
light profiles inferred in the previous search are then used to pass priors.

__Run Times__

In this example we don't explicitly check run-times, for brevity. However, the same rules of thumb we discussed in the
previous tutorial still apply. 

For example, as we add more light profiles to the source model, the likelihood evaluation time will increase. As the
model becomes more complex, search chaining is key to ensuring run times stay lower.

__Model + Search + Analysis + Model-Fit (Search 1)__

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters].
 
 - The source galaxy's light is a parametric linear `Sersic` [6 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
model_1 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=al.mp.Isothermal),
        source=af.Model(al.Galaxy, redshift=1.0, bulge_0=al.lp_linear.Sersic),
    ),
)

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__
"""
analysis_1 = al.AnalysisImaging(dataset=dataset)

search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__mass[sie]__source_x1[bulge]",
    unique_tag=dataset_name,
    n_live=120,
    number_of_cores=1,
)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters: priors initialized from 
 search 1].

 - The source galaxy's light is two parametric linear `Sersic` [12 parameters: first Sersic initialized from 
 search 1].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=17.
"""
model_2 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=result_1.model.galaxies.lens.mass),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge_0=result_1.model.galaxies.source.bulge_0,
            bulge_1=al.lp_linear.Sersic,
        ),
    ),
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]_mass[sie]_source_x2[bulge]",
    unique_tag=dataset_name,
    n_live=120,
    number_of_cores=1,
)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters: priors initialized from 
 search 2].

 - The source galaxy's light is three parametric linear `Sersic` [18 parameters: first two Sersic's initialized from 
 search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=21.
"""
model_3 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=result_2.model.galaxies.lens.mass),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge_0=result_2.model.galaxies.source.bulge_0,
            bulge_1=result_2.model.galaxies.source.bulge_1,
            bulge_2=al.lp_linear.Sersic,
        ),
    ),
)

analysis_3 = al.AnalysisImaging(dataset=dataset)

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]_mass[sie]_source_x3[bulge]",
    unique_tag=dataset_name,
    n_live=140,
    number_of_cores=1,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

 - The lens galaxy's total mass distribution is an `Isothermal` [5 parameters: priors initialized from 
 search 4].

 - The source galaxy's light is four parametric linear `Sersic` [24 parameters: first three Sersic's initialized from 
 search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=29.
"""
model_4 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, mass=result_3.model.galaxies.lens.mass),
        source=af.Model(
            al.Galaxy,
            redshift=1.0,
            bulge_0=result_3.model.galaxies.source.bulge_0,
            bulge_1=result_3.model.galaxies.source.bulge_1,
            bulge_2=result_3.model.galaxies.source.bulge_2,
            bulge_3=al.lp_linear.Sersic,
        ),
    ),
)

analysis_4 = al.AnalysisImaging(dataset=dataset)

search_4 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[4]_mass[sie]_source_x4[bulge]",
    unique_tag=dataset_name,
    n_live=160,
    number_of_cores=1,
)

result_4 = search_4.fit(model=model_4, analysis=analysis_4)

"""
__Wrap Up__

With four light profiles, we were still unable to produce a fit to the source that did not leave residuals. However, I 
actually simulated the lens using a source with four light profiles. A `perfect fit` was therefore somewhere in 
parameter space, but our search unfortunately was unable to locate this.

Lets confirm this, by manually fitting the imaging data with the true input model.

We cannot apply a mask to a dataset that was already masked, so we first reload the imaging from .fits.
"""
dataset = dataset.apply_mask(
    mask=al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light_0=al.lp.Sersic(
        centre=(0.1, 0.1),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.1,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
    light_1=al.lp.Sersic(
        centre=(0.8, 0.6),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.5, angle=30.0),
        intensity=0.2,
        effective_radius=0.3,
        sersic_index=3.0,
    ),
    light_2=al.lp.Sersic(
        centre=(-0.3, 0.6),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.3, angle=120.0),
        intensity=0.6,
        effective_radius=0.5,
        sersic_index=1.5,
    ),
    light_3=al.lp.Sersic(
        centre=(-0.3, -0.3),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=85.0),
        intensity=0.4,
        effective_radius=0.1,
        sersic_index=2.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

true_fit = al.FitImaging(dataset=dataset, tracer=tracer)

fit_plotter = aplt.FitImagingPlotter(fit=true_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

"""
And indeed, we see an improved residual-map, chi-squared-map, and so forth.

If the source morphology is complex, there is no way we chain searches to fit it perfectly. The non-linear parameter 
space simply becomes too complex. For this tutorial, this was true even though our source model could actually fit 
the data perfectly. For  real lenses, the source may be *even more complex* giving us even less hope of getting a 
good fit.

But fear not, **PyAutoLens** has you covered. In chapter 4, we'll introduce a completely new way to model the source 
galaxy, which addresses the problem faced here.
"""
