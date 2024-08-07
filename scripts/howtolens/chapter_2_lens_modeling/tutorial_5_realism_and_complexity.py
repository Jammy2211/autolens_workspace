"""
Tutorial 3: Realism and Complexity
==================================

In the previous two tutorials, we fitted a fairly crude and unrealistic model: the lens's mass was spherical, as was
the source's light. Given most lens galaxies are literally called 'elliptical galaxies' we should probably model their
mass as elliptical! Furthermore, we have completely omitted the lens galaxy's light, which in real observations
outshines the source's light and therefore must be included in the lens model.

In this tutorial, we'll use a more realistic lens model, which consists of the following light and mass profiles:

 - An `Sersic` light profile for the lens galaxy's light [7 parameters].
 - A `Isothermal` mass profile for the lens galaxy's mass [5 parameters].
 - An `ExternalShear` which accounts for additional lensing by other galaxies nearby the lens [2 parameters].
 - An `Exponential` light profile for the source-galaxy's light (this is probably still too simple for most
 strong lenses, but we will worry about that later) [6 parameters].

This lens model has 20 free parameters, meaning that the parameter space and likelihood function it defines has a
dimensionality of N=20. This is over double the number of parameters and dimensions of the models we fitted in the
previous tutorials and in future exercises, we will fit even more complex models with some 30+ parameters.

Therefore, take note, as we make our lens model more realistic, we also make its parameter space more complex, this is
an important concept to keep in mind for the remainder of this chapter!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt
import autofit as af

"""
__Initial Setup__

we'll use new strong lensing data, where:

 - The lens galaxy's light is an `Sersic`.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Exponential`.
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
__Mask__

We'll create and use a 2.5" `Mask2D`, which is slightly smaller than the masks we used in previous tutorials.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
)

dataset = dataset.apply_mask(mask=mask)

"""
When plotted, the lens light`s is clearly visible in the centre of the image.
"""
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Model__

Now lets fit the dataset using a search.

Below, we write the `model` using the **PyAutoFit** concise API, which means that we do not have to specify that
each component of the model is a `Model` object. This is because we are passing the light and mass profiles directly 
to the `Collection` object, which assumes they are `Model` objects.

We will use this consistent API throughout the chapter, so you should get used to it!
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=0.5,
            bulge=al.lp.Sersic,
            mass=al.mp.Isothermal,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.ExponentialCore),
    ),
)

"""
__Search + Analysis__

We set up `Nautilus` as we did in the previous tutorial, however given the increase in model complexity we'll use
a higher `n_live` value of 150 to ensure we sample the complex parameter space efficiently.
"""
search = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_2"),
    name="tutorial_3_realism_and_complexity",
    unique_tag=dataset_name,
    n_live=200,
    number_of_cores=1,
)

analysis = al.AnalysisImaging(dataset=dataset)

"""
__Run Time__

The run time of the `log_likelihood_function` is a little bit more than previous tutorials, because we added more
light and mass profiles to the model. It is the increased number of parameters that increases the expected run time 
more.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
Run the non-linear search.
"""
print(
    "The non-linear search has begun running - checkout the autolens_workspace/output/howtolens/chapter_2/tutorial_3_realism_and_complexity"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result = search.fit(model=model, analysis=analysis)

print("Search has finished run - you may now continue the notebook.")

"""
__Result__

Inspection of the `info` summary of the result suggests the model has gone to reasonable values.
"""
print(result.info)

"""
And lets look at how well the model fits the imaging data, which as we are used to fits the data brilliantly!
"""
fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Global and Local Maxima__

Up to now, all our non-linear searches have successfully found lens models that provide visibly good fits to the data, 
minimizing residuals and inferring high log likelihood values. These optimal solutions, known as 'global maxima,' 
correspond to the highest likelihood regions across the entire parameter space. In other words, no other lens model 
in the parameter space would yield a higher likelihood. This is the ideal model we always aim to infer.

However, non-linear searches do not always locate these global maxima. They may instead infer 'local maxima' 
solutions, which have high log likelihood values relative to nearby models in the parameter space, but 
significantly lower log likelihoods compared to the true global maxima situated elsewhere. 

Why might a non-linear search end up at these local maxima? As previously discussed, the search iterates through 
many models, focusing more on regions of the parameter space where previous guesses yielded higher likelihoods. 
The search gradually 'converges' around any solution with a higher likelihood than surrounding models. If the 
search is not thorough enough, it may settle on a local maxima, appearing to offer a high likelihood 
relative to nearby models but failing to reach the global maxima.

Inferring local maxima is a failure of our non-linear search, and it's something we want to avoid. To illustrate 
this, we can intentionally infer a local maxima by reducing the number of live points (`n_live`) Nautilus uses to 
map out the parameter space. By using very few live points, the initial search over the parameter space has a low 
probability of approaching the global maxima, thus it converges on a local maxima.
"""
search = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_2"),
    name="tutorial_3_realism_and_complexity__local_maxima",
    unique_tag=dataset_name,
    n_live=75,
    number_of_cores=1,
)

print(
    "The non-linear search has begun running - checkout the autolens_workspace/output/3_realism_and_complexity"
    " folder for live output of the results, images and lens model."
    " This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

"""
__Run Time__

Due to the decreased number of live points, the estimate of 10000 iterations per free parameter is now a significant
overestimate. The actual run time of the model-fit will be much less than this.
"""
run_time_dict, info_dict = analysis.profile_log_likelihood_function(
    instance=model.random_instance()
)

print(f"Log Likelihood Evaluation Time (second) = {run_time_dict['fit_time']}")
print(
    "Estimated Run Time Upper Limit (seconds) = ",
    (run_time_dict["fit_time"] * model.total_free_parameters * 10000)
    / search.number_of_cores,
)

"""
Run the non-linear search.
"""
result_local_maxima = search.fit(model=model, analysis=analysis)

print("Search has finished run - you may now continue the notebook.")

"""
__Result__

Inspection of the `info` summary of the result suggests certain parameters have gone to different values to the fit
performed above.
"""
print(result_local_maxima.info)

"""
Lats look at the fit to the `Imaging` data, which is clearly worse than our original fit above.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result_local_maxima.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
Finally, just to be sure we hit a local maxima, lets compare the maximum log likelihood values of the two results 

The local maxima value is significantly lower, confirming that our non-linear search simply failed to locate lens 
models which fit the data better when it searched parameter space.
"""
print("Likelihood of Global Model:")
print(result.max_log_likelihood_fit.log_likelihood)
print("Likelihood of Local Model:")
print(result_local_maxima.max_log_likelihood_fit.log_likelihood)

"""
__Wrap Up__

In this example, we intentionally made our non-linear search fail, by using so few live points it had no hope of 
sampling parameter space thoroughly. For modeling real lenses we wouldn't do this intentionally, but the risk of 
inferring a local maxima is still very real, especially as we make our lens model more complex.

Lets think about *complexity*. As we make our lens model more realistic, we also made it more complex. For this 
tutorial, our non-linear parameter space went from 7 dimensions to 18. This means there was a much larger *volume* of 
parameter space to search. As this volume grows, there becomes a higher chance that our non-linear search gets lost 
and infers a local maxima, especially if we don't set it up with enough live points!

At its core, lens modeling is all about learning how to get a non-linear search to find the global maxima region of 
parameter space, even when the lens model is complex. This will be the main theme throughout the rest of this chapter
and is the main subject of chapter 3.

In the next exercise, we'll learn how to deal with failure and begin thinking about how we can ensure our non-linear 
search finds the global-maximum log likelihood solution. First, think about the following:

 1) When you look at an image of a strong lens, do you get a sense of roughly what values certain lens model 
 parameters are?
    
 2) The non-linear search failed because parameter space was too complex. Could we make it less complex, whilst 
 still keeping our lens model fairly realistic?
    
 3) The source galaxy in this example had only 7 non-linear parameters. Real source galaxies may have multiple 
 components (e.g. a disk, bulge, bar, star-forming knot) and there may even be more than 1 source galaxy! Do you 
 think there is any hope of us navigating a parameter space if the source contributes 20+ parameters by itself?
"""
