# %%
"""
__Linking Phases__

So, we've learnt that if our parameter space is too complex, our non-linear search might fail to find the global
maximum solution. However, we also learnt how to ensure this doesn't happen, by:

    1) Tuning our priors to the strong lens we're fitting.
    2) Making our lens model less complex.
    3) Searching non-linear parameter space for longer.

However, each of the above approaches has disadvantages. The more we tune our priors, the less we can generalize our
analysis to a different strong lens. The less complex we make our model, the less realistic it is. And if we rely too
much on searching parameter space for longer, we could end up with phase's that take days, weeks or months to run.

In this exercise, we're going to combine these 3 approaches so that we can fit complex and realistic lens models in a
way that that can be generalized to many different strong lenses. To do this, we'll run 2 phases, and link the lens
model inferred in the first phase to the priors of the second phase's lens model.

Our first phase will make the same light-traces-mass assumption we made in the previous tutorial. We saw that this
gives a reasonable lens model. However, we'll make a couple of extra simplifying assumptions, to really try and bring
our lens model complexity down and get the non-linear search running fast.

The model we infer above will therefore be a lot less realistic. But it doesn't matter, because in the second phase
we're going to relax these assumptions and get back our more realistic lens model. The beauty is that, by running the
first phase, we can use its results to tune the priors of our second phase. For example:

1) The first phase should give us a pretty good idea of the lens galaxy's light and mass profiles, for example its
   intensity, effective radius and einstein radius.

2) It should also give us a pretty good fit to the lensed source galaxy. This means we'll already know where in
   source-plane its is located and what its intensity and effective are.
"""

# %%
#%matplotlib inline

from autoconf import conf
import autolens as al
import autolens.plot as aplt
import autofit as af

# %%
"""
You need to change the path below to the chapter 1 directory.
"""

# %%
workspace_path = "/path/to/user/autolens_workspace/howtolens"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"

# %%
"""
We'll use the same strong lensing data as the previous tutorial, where:

    - The lens galaxy's _LightProfile_ is an _EllipticalSersic_.
    - The lens galaxy's _MassProfile_ is an *EllipticalIsothermal_.
    - The source galaxy's _LightProfile_ is an _EllipticalExponential_.
"""

# %%
dataset_label = "chapter_2"
dataset_name = "lens_sersic_sie__source_exp"
dataset_path = f"{workspace_path}/howtolens/dataset/{dataset_label}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.1,
)

# %%
"""
We'll create and use a smaller 2.0" _Mask_ again.
"""

# %%
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=2.0
)

# %%
"""
When plotted, the lens light's is clearly visible in the centre of the image.
"""

# %%
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
Like in the previous tutorial, we use a_PhaseSettingsImaging_ object to specify our model-fitting procedure uses a 
regular _Grid_.
"""

# %%
settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=2)

# %%
"""
As we've eluded to before, one can look at an image and immediately identify the centre of the lens galaxy. It's 
that bright blob of light in the middle! Given that we know we're going to make the lens model more complex in the 
next phase, lets take a more liberal approach than before and fix the lens centre to (y,x) = (0.0", 0.0").
"""

# %%
lens = al.GalaxyModel(
    redshift=0.5, light=al.lp.EllipticalSersic, mass=al.mp.EllipticalIsothermal
)

source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalExponential)

# %%
"""
You haven't actually seen a line like this one before. By setting a parameter to a number (and not a prior) it is be 
removed from non-linear parameter space and always fixed to that value. Pretty neat, huh?
"""

# %%
lens.light.centre_0 = 0.0
lens.light.centre_1 = 0.0
lens.mass.centre_0 = 0.0
lens.mass.centre_1 = 0.0

# %%
"""
Lets use the same approach of making the ellipticity of the mass trace that of the light.
"""
lens.mass.elliptical_comps_0 = lens.light.elliptical_comps_0
lens.mass.elliptical_comps_1 = lens.light.elliptical_comps_1

# %%
"""
Now, you might be thinking, doesn't this prevent our phase from generalizing to other strong lenses? What if the 
centre of their lens galaxy isn't at (0.0", 0.0")?

Well, this is true if our dataset reduction centres the lens galaxy somewhere else. But we get to choose where we 
centre it when we make the image. Therefore, I'd recommend you always centre the lens galaxy at the same location, 
and (0.0", 0.0") seems the best choice!

We also discussed that the Sersic index of most lens galaxies is around 4. Lets fix it to 4 this time.
"""

# %%
lens.light.sersic_index = 4.0

# %%
"""
Now lets create the phase.
"""

# %%
phase_1 = al.PhaseImaging(
    phase_name="phase_t5_linking_phases_1",
    settings=settings,
    galaxies=dict(lens=lens, source=source),
    search=af.DynestyStatic(n_live_points=40, sampling_efficiency=0.8, evidence_tolerance=100.0),
)

# %%
"""
Lets run the phase, noting that our liberal approach to reducing the lens model complexity has reduced it to just 
11 parameters. (The results are still preloaded for you, but feel free to run it yourself, its fairly quick).
"""

# %%
print(
    "Dynesty has begun running - checkout the workspace/output/5_linking_phases"
    "folder for live output of the results, images and lens model."
    "This Jupyter notebook cell with progress once Dynesty has completed - this could take some time!"
)

phase_1_result = phase_1.run(dataset=imaging, mask=mask)

print("Dynesty has finished run - you may now continue the notebook.")

# %%
"""
And indeed, we get a reasonably good model and fit to the data - in a much shorter space of time!
"""

# %%
aplt.FitImaging.subplot_fit_imaging(fit=phase_1_result.max_log_likelihood_fit)

# %%
"""
Now all we need to do is look at the results of phase 1 and tune our priors in phase 2 to those results. Lets setup 
a custom phase that does exactly that.

GaussianPriors are a nice way to do this. They tell the non-linear search where to look, but leave open the 
possibility that there might be a better solution nearby. In contrast, UniformPriors put hard limits on what values a 
parameter can or can't take. It makes it more likely we'll accidently cut-out the global maxima solution.
"""

# %%
lens = al.GalaxyModel(
    redshift=0.5, light=al.lp.EllipticalSersic, mass=al.mp.EllipticalIsothermal
)
source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalExponential)

# %%
"""
What I've done below is looked at the results of phase 1 and manually specified a prior for every parameter. If a 
parameter was fixed in the previous phase, its prior is based around the previous value. Don't worry about the sigma 
values for now, I've chosen values that I know will ensure reasonable sampling, but we'll cover this later.
"""

# %%

## LENS LIGHT PRIORS ###

lens.light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.light.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.light.elliptical_comps_0 = af.GaussianPrior(mean=0.33333, sigma=0.15)
lens.light.elliptical_comps_1 = af.GaussianPrior(mean=0.0, sigma=15.0)
lens.light.intensity = af.GaussianPrior(mean=0.02, sigma=0.01)
lens.light.effective_radius = af.GaussianPrior(mean=0.62, sigma=0.2)
lens.light.sersic_index = af.GaussianPrior(mean=4.0, sigma=2.0)

### LENS MASS PRIORS ###

lens.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
lens.mass.elliptical_comps_0 = af.GaussianPrior(mean=0.33333, sigma=0.15)
lens.mass.elliptical_comps_1 = af.GaussianPrior(mean=0.0, sigma=15.0)
lens.mass.einstein_radius = af.GaussianPrior(mean=0.8, sigma=0.1)

### SOURCE LIGHT PRIORS ###

source.light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
source.light.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)
source.light.axis_ratio = af.GaussianPrior(mean=0.8, sigma=0.1)
source.light.phi = af.GaussianPrior(mean=90.0, sigma=10.0)
source.light.intensity = af.GaussianPrior(mean=0.14, sigma=0.05)
source.light.effective_radius = af.GaussianPrior(mean=0.12, sigma=0.2)

# %%
"""
Lets setup and run the phase. As expected, it gives us the correct lens model. However, it does so significantly faster 
than we're used to - I didn't have to edit the config files to get this phase to run fast!
"""

# %%
phase_2 = al.PhaseImaging(
    phase_name="phase_t5_linking_phases_2",
    settings=settings,
    galaxies=dict(lens=lens, source=source),
    search=af.DynestyStatic(n_live_points=40, sampling_efficiency=0.5, evidence_tolerance=100.0),
)

print(
    "Dynesty has begun running - checkout the workspace/output/5_linking_phases"
    "folder for live output of the results, images and lens model."
    "This Jupyter notebook cell with progress once Dynesty has completed - this could take some time!"
)

phase_2_results = phase_2.run(dataset=imaging, mask=mask)

print("Dynesty has finished run - you may now continue the notebook.")

# %%
"""
Look at that, the right lens model, again!
"""

# %%
aplt.FitImaging.subplot_fit_imaging(fit=phase_2_results.max_log_likelihood_fit)

# %%
"""
Our choice to link two phases together was a huge success. We managed to fit a complex and realistic model, but were 
able to begin by making simplifying assumptions that eased our search of non-linear parameter space. We could apply 
phase 1 to pretty much any strong lens and therefore get ourselves a decent lens model with which to tune phase 2's 
priors.

You're probably thinking though that there is one huge, giant, glaring flaw in all of this that I've not mentioned. 
Phase 2 can't be generalized to another lens - it's priors are tuned to the image we fitted. If we had a lot of lenses, 
we'd have to write a new phase_2 for every single one. This isn't ideal, is it?

Fotunately, we can pass priors in PyAutoLens without specifying the specific values, using what we call promises. The
code below sets up phase_2 with priors fully linked, but without specifying each individual prior!
"""

# %%
phase_2 = al.PhaseImaging(
    phase_name="phase_t5_linking_phases_2",
    settings=settings,
    galaxies=dict(
        lens=phase_1_result.model.galaxies.lens,
        source=phase_1_result.model.galaxies.source
    ),
    search=af.DynestyStatic(n_live_points=40, sampling_efficiency=0.5, evidence_tolerance=100.0),
)

# %%
"""
Above, the result of our phase 1 fit are used to set up the priors on the lens and source _GalaxyModel_ objects. This 
code *can* be generalized to any lens system!

In chapter 3, we'll cover 'pipelines'. A pipeline comprises a set of phases that are linked together, allowing us to 
start with a simple, easy-to-fit lens model, and gradually makes it more complex. Crucially, as the pipeline runs, 
we 'feed' the results of previous phases through the pipeline, allowing us to tune our priors automatically, 
in a way that can be applied generically to any strong lens.
"""
