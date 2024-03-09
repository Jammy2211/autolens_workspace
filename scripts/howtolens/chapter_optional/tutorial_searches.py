"""
Tutorial: Alternative Searches
==============================

Up to now, we've always used the non-linear search Nautilus and not considered the input parameters that control its
sampling. In this tutorial, we'll consider how we can change these setting to balance finding the global maxima
solution with fast run time.

We will also discuss other types of non-linear searches, such as MCMC and optimizers, which we can use to perform lens
modeling. So far, we have no found any of these alternatives to give anywhere near as robust and efficient results as
Nautilus, and we recommend users use Nautilus unless they are particularly interested in investigating different
model-fitting techniques.
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
we'll use new strong lensing data, where:

 - The lens galaxy's light is an `Sersic`.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.
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
we'll create and use a smaller 2.0" `Mask2D` again.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.6
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset, visuals_2d=aplt.Visuals2D(mask=mask)
)
dataset_plotter.subplot_dataset()

"""
__Nested Sampling__

Lets first perform the model-fit using Nautilus, but look at different parameters that control how long it takes to run. 
We'll therefore discuss in a bit more detail how Nautilus works, but still keep this description conceptually simple  
and avoid technical terms and jargon. For a complete description of Nautilus you should check out the Nautilus 
publication `https://arxiv.org/abs/2306.16923`.

nlive:

Nautilus is a `nested sampling` algorithm. As we described in chapter 2, it throws down a set of `live points` in 
parameter space, where each live point corresponds to a lens model with a given set of parameters. These points are
initially distributed according to our priors, hence why tuning our priors allows us to sample parameter space faster.
 
The number of live points is set by the parameter `n_live`. More points provide a more thorough sampling of 
parameter space, increasing the probability that we locate the global maxima solution. Therefore, if you think your 
model-fit has gone to a local maxima, you should try increasing `n_live`. The downside of this is Nautilus will 
take longer to sample parameter space and converge on a solution. Ideally, we will use as few live points as possible 
to locate the global maxima as quickly as possible.

f_live:

A nested sampling algorithm estimates the *Bayesian Evidence* of the model-fit, which is quantity the non-linear 
search algorithms we introduce later do not. The Bayesian evidence quantifies how well the lens model as a whole fits
the data, following a principle called Occam's Razor (`https://simple.wikipedia.org/wiki/Occam%27s_razor`). This 
penalizes models for being more complex (e.g. more parameters) and requires that their additional complexity improve 
their overall fit to the data compared to a simpler model. By computing the comparing the Bayesian evidence of 
different models one can objectively choose the lens model that best fits the data.

A nested sampling algorithm stops sampling when it estimates that continuing sampling will not increase the Bayesian 
evidence (called the `log_evidence`) by more than the `f_live`. As Nautilus progresses and converges on the
solution, the rate of increase of the estimated Bayesian evidence slows down. Therefore, higher `f_live`s 
mean Nautilus terminate sooner.
    
A high `f_live` will make the errors estimated on every parameter unreliable and its value must be kept 
below 0.8 for reliable error estimates. However, when chaining searches, we typically *do not care* about the errors 
in the first search, therefore setting a high evidence tolerance can be an effective means to make Nautilus converge
faster (we'll estimate reliable errors in the second search when the `f_live is 0.8 or less). 

Lets perform two fits, where:

 - One has many live points and a higher evidence tolerance, causing the non-linear search to
 take a longer time to run.
      
 - One has few live points, a high sampling efficiency and evidence tolerance, causing the non-linear search to
 converge and end quicker.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy, redshift=0.5, bulge=al.lp.Sersic, mass=al.mp.Isothermal
        ),
        source=af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic),
    ),
)

search = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_optional"),
    name="tutorial_searches_slow",
    unique_tag=dataset_name,
    n_live=400,
)

analysis = al.AnalysisImaging(dataset=dataset)

print(
    "The non-linear search has begun running - checkout the workspace/output"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_slow = search.fit(model=model, analysis=analysis)

"""
Lets check that we get a good model and fit to the data.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result_slow.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
We can use the result to tell us how many iterations Nautilus took to convergence on the solution.
"""
print("Total Nautilus Iterations (If you skip running the search, this is ~ 500000):")
print(result_slow.samples.total_samples)

"""
Now lets run the search with fast settings, so we can compare the total number of iterations required.
"""
search = af.Nautilus(
    path_prefix=path.join("howtolens", "chapter_2"),
    name="tutorial_searches_fast",
    unique_tag=dataset_name,
    n_live=75,
)

print(
    "The non-linear search has begun running - checkout the workspace/output"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_fast = search.fit(model=model, analysis=analysis)

print("Search has finished run - you may now continue the notebook.")

"""
Lets check that this search, despite its faster sampling settings, still gives us the global maxima solution.
"""
fit_plotter = aplt.FitImagingPlotter(fit=result_fast.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
And now lets confirm it uses significantly fewer iterations.
"""
print("Total Nautilus Iterations:")
print("Slow settings: ~500000")
print(result_slow.samples.total_samples)
print("Fast settings: ", result_fast.samples.total_samples)

"""
__Optimizers__

Nested sampling algorithms like Nautilus provides the errors on all of the model parameters, by fully mapping out all 
of the high likelihood regions of parameter space. This provides knowledge on the complete *range* of models that do 
and do not provide high likelihood fits to the data, but takes many extra iterations to perform. If we require precise 
error estimates (perhaps this is our final lens model fit before we publish the results in a paper), these extra
iterations are acceptable. 

However, we often don't care about the errors. For example, in the previous tutorial when chaining searches, the only 
result we used from the fit performed in the first search was the maximum log likelihood model, omitting the errors
entirely! Its seems wasteful to use a nested sampling algorithm like Nautilus to map out the entirity of parameter
space when we don't use this information! 

There are a class of non-linear searches called `optimizers`, which seek to optimize just one thing, the log 
likelihood. They want to find the model that maximizes the log likelihood, with no regard for the errors, thus not 
wasting time mapping out in intricate detail every facet of parameter space. Lets see how much faster we can find a 
good fit to the lens data using an optimizer.

we'll use the `Particle Swarm Optimizer` PySwarms. Conceptually this works quite similar to Nautilus, it has a set of 
points in parameter space (called `particles`) and it uses their likelihoods to determine where it thinks the higher
likelihood regions of parameter space are. 

Unlike Nautilus, this algorithm requires us to specify how many iterations it should perform to find the global 
maxima solutions. Here, an iteration is the number of samples performed by every particle, so the total number of
iterations is n_particles * iters. Lets try a total of 50000 iterations, a factor 10 less than our Nautilus runs above. 

In our experience, pyswarms is ineffective at initializing a lens model and therefore needs a the initial swarm of
particles to surround the highest likelihood lens models. We set this starting point up below by manually inputting 
`GaussianPriors` on every parameter, where the centre of these priors is near the true values of the simulated lens data.

Given this need for a robust starting point, PySwarms is only suited to model-fits where we have this information. It may
therefore be useful when performing lens modeling search chaining (see HowToLens chapter 3). However, even in such
circumstances, we have found that is often unrealible and often infers a local maxima.
"""
lens_bulge = af.Model(al.lp.Sersic)
lens_bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
lens_bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
lens_bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
lens_bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
lens_bulge.intensity = af.GaussianPrior(mean=1.0, sigma=0.3)
lens_bulge.effective_radius = af.GaussianPrior(mean=0.8, sigma=0.2)
lens_bulge.sersic_index = af.GaussianPrior(mean=4.0, sigma=1.0)

mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
mass.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
mass.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
mass.einstein_radius = af.GaussianPrior(mean=1.4, sigma=0.4)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
shear.gamma_2 = af.GaussianPrior(mean=0.0, sigma=0.1)

bulge = af.Model(al.lp.Sersic)
bulge.centre.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.centre.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.ell_comps.ell_comps_0 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.ell_comps.ell_comps_1 = af.GaussianPrior(mean=0.0, sigma=0.3)
bulge.intensity = af.GaussianPrior(mean=0.3, sigma=0.3)
bulge.effective_radius = af.GaussianPrior(mean=0.2, sigma=0.2)
bulge.sersic_index = af.GaussianPrior(mean=1.0, sigma=1.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.PySwarmsLocal(
    path_prefix=path.join("howtolens", "chapter_optional"),
    name="tutorial_searches_pso",
    unique_tag=dataset_name,
    n_particles=50,
    iters=1000,
)

print(
    "The non-linear search has begun running - checkout the workspace/output"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_pso = search.fit(model=model, analysis=analysis)

print("PySwarms has finished run - you may now continue the notebook.")

fit_plotter = aplt.FitImagingPlotter(fit=result_pso.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
In our experience, the parameter spaces fitted by lens models are too complex for `PySwarms` to be used without a lot
of user attention and care and careful setting up of the initialization priors, as shown above.

__MCMC__

For users familiar with Markov Chain Monte Carlo (MCMC) non-linear samplers, PyAutoFit supports the non-linear
searches `Emcee` and `Zeus`. Like PySwarms, these also need a good starting point, and are generally less effective at 
lens modeling than Nautilus. 

I've included an example runs of Emcee and Zeus below, where the model is set up using `UniformPriors` to give
the starting point of the MCMC walkers. 
"""
lens_bulge = af.Model(al.lp.Sersic)
lens_bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
lens_bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
lens_bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
lens_bulge.intensity = af.UniformPrior(lower_limit=0.5, upper_limit=1.5)
lens_bulge.effective_radius = af.UniformPrior(lower_limit=0.2, upper_limit=1.6)
lens_bulge.sersic_index = af.UniformPrior(lower_limit=3.0, upper_limit=5.0)


mass = af.Model(al.mp.Isothermal)
mass.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
mass.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
mass.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
mass.einstein_radius = af.UniformPrior(lower_limit=1.0, upper_limit=2.0)

shear = af.Model(al.mp.ExternalShear)
shear.gamma_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
shear.gamma_2 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

bulge = af.Model(al.lp.Sersic)
bulge.centre.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.centre.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)
bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=-0.3, upper_limit=0.3)
bulge.intensity = af.UniformPrior(lower_limit=0.1, upper_limit=0.5)
bulge.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.4)
bulge.sersic_index = af.UniformPrior(lower_limit=0.5, upper_limit=2.0)

lens = af.Model(al.Galaxy, redshift=0.5, mass=mass, shear=shear)
source = af.Model(al.Galaxy, redshift=1.0, bulge=bulge)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

search = af.Zeus(
    path_prefix=path.join("howtolens", "chapter_2"),
    name="tutorial_searches_zeus",
    unique_tag=dataset_name,
    nwalkers=50,
    nsteps=1000,
)

print(
    "Zeus has begun running - checkout the workspace/output"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_zeus = search.fit(model=model, analysis=analysis)

print("Zeus has finished run - you may now continue the notebook.")

fit_plotter = aplt.FitImagingPlotter(fit=result_zeus.max_log_likelihood_fit)
fit_plotter.subplot_fit()


search = af.Emcee(
    path_prefix=path.join("howtolens", "chapter_2"),
    name="tutorial_searches_emcee",
    unique_tag=dataset_name,
    nwalkers=50,
    nsteps=1000,
)

print(
    "The non-linear search has begun running - checkout the workspace/output"
    "  folder for live output of the results, images and lens model."
    "  This Jupyter notebook cell with progress once search has completed - this could take some time!"
)

result_emcee = search.fit(model=model, analysis=analysis)

print("The search has finished run - you may now continue the notebook.")

fit_plotter = aplt.FitImagingPlotter(fit=result_emcee.max_log_likelihood_fit)
fit_plotter.subplot_fit()
