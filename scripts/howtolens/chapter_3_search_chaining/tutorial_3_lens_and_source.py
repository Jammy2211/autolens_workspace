"""
Tutorial 3: Lens and Source
===========================

In this tutorial, we demonstrate search chaining using three searches to fit strong lens `Imaging` which includes the
lens galaxy's light.

The crucial point to note is that for many lenses the lens galaxy's light can be fitted and subtracted reasonably 
well before we attempt to fit the source galaxy. This makes sense, as fitting the lens's light (which is an elliptical
blob of light in the centre of the imaging) looks nothing like the source's light (which is a ring of light)! Formally,
we would say that these two model components (the lens's light and source's light) are not covariate.

So, as a newly trained lens modeler, what does the lack of covariance between these parameters make you think?
Hopefully, you're thinking, why should I bother fitting the lens and source galaxy simultaneously? Surely we can
find the right regions of non-linear parameter space by fitting each separately first? This is what we're going to do
in this tutorial, using a pipeline composed of a modest 3 searches:

 1) Fit the lens galaxy's light, ignoring the source.
 2) Fit the source-galaxy's light (and therefore lens galaxy's mass), ignoring the len`s light.
 3) Fit both simultaneously, using these results to initialize our starting location in parameter space.

Of course, given that we do not care for the errors in searches 1 and 2, we will set up our non-linear search to
perform sampling as fast as possible!

__Dated Tutorial__

This example tutorial was written ~4 years ago, when **PyAutoLens** was in its infancy and had a number of limitations:

 - The non-linear search used MultiNest or dynesty, which were less reliable (e.g. more likely to infer a local maxima
  for complex lens models) and less efficient than Nautilus.

 - Linear light profiles and techniques like a Multi-Gaussian Expansion were not available.

With all the new features added to **PyAutoLens** since, we no longer recommend that one breaks down the fitting of
the lens and source galaxy's light into separate searches, as perform in this search chaining example. Instead, we
would recommend you fit the lens and source simultaneously, using linear light profiles to make the model simpler
or a Multi-Gaussian Expansion.

However, the example is still useful for demonstrating the core concepts of search chaining, which is still vital
for fitting complex lens model. Therefore, we recommend you still read through this tutorial and try to get a good
understanding of how search chaining works, but bear in mind that the example is a little dated and we now recommend
you fit the lens and source simultaneously!
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

we'll use strong lensing data, where:

 - The lens galaxy's light is an `Sersic`.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Exponential`.
 
This image was fitted throughout chapter 2.
"""
dataset_name = "lens_sersic"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

"""
__Paths__

All three searches will use the same `path_prefix`, so we write it here to avoid repetition.
"""
path_prefix = path.join("howtolens", "chapter_3", "tutorial_3_lens_and_source")

"""
__Masking (Search 1)__

We need to choose our mask for the analysis. Given we are only fitting the lens light we have two options: 

 - A circular mask that does not remove the source's light from the fit, assuming the lens light model will still be 
 sufficiently accurate to reveal the source in the second search.
 - An 'anti-annular' mask that removes the source's ring of light.

In this example, we will use the anti-annular mask to demonstrate that we can change the mask used by each search in a 
chain of non-linear searches.
"""
mask = al.Mask2D.circular_anti_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.8,
    outer_radius=2.2,
    outer_radius_2=3.0,
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's light is a parametric linear `Sersic` bulge [6 parameters].
 
 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=6.

__Notes__

We use linear light profiles througout this script, given that the model is quite complex and this helps
simplify it.
"""
model_1 = af.Collection(
    galaxies=af.Collection(lens=af.Model(al.Galaxy, redshift=0.5, bulge=al.lp.Sersic)),
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
    name="search[1]_light[bulge]",
    unique_tag=dataset_name,
    n_live=75,
    number_of_cores=4,
)

"""
__Run Time__

It is good practise to always check the `log_likelihood_function` run time before starting the non-linear search.  
It will be similar to the value we saw in the previous chapter.
"""
run_time_dict, info_dict = analysis_1.profile_log_likelihood_function(
    instance=model_1.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model_1.total_free_parameters * 10000)
    / search_1.number_of_cores,
)

"""
Run the search.
"""

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_1.info)

"""
__Masking (Search 2)__

Search 2 we are only fitting the source's light, thus we can apply an annular mask that removes regions of the
image that contained only the lens's light.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=0.6,
    outer_radius=2.4,
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

Search 2 fits a lens model where:

 - The lens galaxy's light is a linear `Sersic` bulge [Parameters fixed to results of search 1].
 
 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric linear `Sersic` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=13.

Search 2, we fit the source-`galaxy's light and fix the lens light model to the model inferred in search 1, 
ensuring the image we has the foreground lens subtracted. We do this below by passing the lens light as an `instance` 
object.

By passing an `instance`, we are telling **PyAutoLens** that we want it to pass the maximum log likelihood result of 
that search and use those parameters as fixed values in the model. The model parameters passed as an `instance` are not 
free parameters fitted for by the non-linear search, thus this reduces the dimensionality of the non-linear search 
making model-fitting faster and more reliable. 
     
Thus, search 2 includes the lens light model from search 1, but it is completely fixed during the model-fit!

We also use the centre of the `bulge` to initialize the priors on the lens's `mass`.
"""
mass = af.Model(al.mp.Isothermal)
mass.centre_0 = result_1.model.galaxies.lens.bulge.centre_0
mass.centre_1 = result_1.model.galaxies.lens.bulge.centre_1

model_2 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=result_1.instance.galaxies.lens.bulge,
            mass=mass,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    ),
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]_mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=4,
)

"""
__Run Time__

The run-time of the fit should be noticeably faster than the previous search, but because the smaller mask means the
likelihood function is evaluated faster and because prior passing ensures the search samples parameter space faster.
"""
run_time_dict, info_dict = analysis_2.profile_log_likelihood_function(
    instance=model_2.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model_2.total_free_parameters * 10000)
    / search_2.number_of_cores,
)

"""
Run the search.
"""
result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 2)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_2.info)

"""
__Masking (Search 3)__

Search 3 we fit the lens and source, therefore we will use a large circular mask.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

Search 3 fits a lens model where:

 - The lens galaxy's light is a linear `Sersic` bulge [6 Parameters: priors initialized from search 1].
 
 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters: priors
 initialized from search 2].
 
 - The source galaxy's light is a parametric linear `Sersic` [6 parameters: priors initialized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=23.

There isn't a huge amount to say about this search, we have initialized the priors on all of our models parameters
and the only thing that is left to do is fit for all model components simultaneously, with slower Nautilus settings
that will give us more accurate parameter values and errors.
"""
model_3 = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=result_1.model.galaxies.lens.bulge,
            mass=result_2.model.galaxies.lens.mass,
        ),
        source=af.Model(
            al.Galaxy, redshift=1.0, bulge=result_2.model.galaxies.source.bulge
        ),
    ),
)

analysis_3 = al.AnalysisImaging(dataset=dataset)

search_3 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[3]_light[bulge]_mass[sie]_source[bulge]",
    unique_tag=dataset_name,
    n_live=150,
    number_of_cores=4,
)

result_3 = search_3.fit(model=model_3, analysis=analysis_3)

"""
__Result (Search 3)__

The final results are summarized in the `info` attribute.
"""
print(result_3.info)

"""
__Wrap Up__

And there we have it, a sequence of searches that breaks modeling the lens and source galaxy into 3 simple searches. 
This approach is much faster than fitting the lens and source simultaneously from the beginning. Instead of asking you 
questions at the end of this chapter`s tutorials, I'm going to ask questions which I then answer. This will hopefully 
get you thinking about how to approach pipeline writing.

 1) Can this pipeline really be generalized to any lens? Surely the radii of the masks depends on the lens and source 
 galaxies?

Whilst this is true, we chose mask radii above that are `excessive` and masks out a lot more of the image than just 
the source (which, in terms of run-time, is desirable). Thus, provided you know the Einstein radius distribution of 
your lens sample, you can choose mask radii that will masks out every source in your sample adequately (and even if 
some of the source is still there, who cares? The fit to the lens galaxy will be okay).

However, the template pipelines provided on the `autolens_workspace` simply use circular masks for every search and do
not attempt to use different masks for the lens light fit and source fit. This is to keep things simple (at the expense
of slower run times). It is up to you if you want to adapt these scripts to try and use more specific masking strategies.

__Dated Tutorial__

In fact, we now strongly recommend that you do not change masks between each search when using search chaining. 
This is because it is very fiddly, and can waste a lot of your time refining masks to ensure they are suitable for
each lens. We recommend you always just use a large circular mask which is big enough to include the entire lens and 
source of all lenses in your sample. This will save you a lot of time and means lens modeling can be automated much
easier.

Building on the discussion above, a known limitation of using a pipeline which fits the lens light first, then the
source, is that it will do a poor job deblending the lens and source light if the Einstein radius is low. This often
leads the mass model to infer incorrect solutions which fit residuals from the lens light subtraction.

This is why, given all the improvements to autolens, we now recommend that you do not use this pipeline and instead
always begin by fitting the lens and source simultaneously. This can use linear light profiles of a Multi-Gaussian
Expansion. 
"""
