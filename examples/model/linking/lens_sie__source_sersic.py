# %%
"""
__Example: Linking Phases__

In the 'beginner' examples all model-fits were performed using one phase, which composed the lens model using one
parametrization and performed the model-fit using one non-linear search. In the 'linking' examples we break the
model-fitting procedure down into multiple phases, linking the results of the initial phases to subsequent phases.
This allows us to guide the model-fitting procedure as to where it should look in parameter space for the
highest log-likelihood models.

When linking phases:

    - The earlier phases fit simpler model parameterizatons than the later phases, providing them with a less complex
      non-linear parameter space that can be sampled more efficiently and with a reduced chance of inferring an
      incorrect local maxima solution.

    - The earlier phases use non-linear search techniques that only seek to maximize the log likelihood and do not
      precisely quantify the errors on every parameter, whereas the latter phases do. This means we can 'initialize'
      a model-fit very quickly and only spend more computational time estimating errors in the final phase when we
      actually require them.

    - The earlier phases can use the _PhaseSettingsImaging_ object to augment the data or alter the fitting-procedure
      in ways that speed up the computational run time. These may impact the quality of the model-fit overall, but they
      can be reverted to the more accurate but more computationally expense setting in the final phases.

Prior linking is crucial the PyAutoLens pipelines found in the folder 'autolens_workspace/pipelines'. These example
provide a conceptual overview of why prior linking is used and an introduction to the API used to do so. More details
on prior linking can be found in Chapter 2 of the HowToLens lectures, specifically 'tutorial_5_linking_phases.py'.
"""

# %%
"""
This example scripts show a simple example of prior linking, where we fit imaging of a strong lens system where:

    - The lens galaxy's _LightProfile_ is omitted (and is not present in the simulated data.
    - The lens galaxy's _MassProfile_ is fitted with an _EllipticalIsothermal_.
    - The source galaxy's _LightProfile_ is fitted with an _EllipticalSersic_.

As discussed below, the first phase is set up to provide as fast a model-fit as possible without accurately quantifying
the errors on every parameter, whereas the second phase sacrifices this run-speed for accuracy. 
"""

# %%
# %matplotlib inline

import numpy as np

from autoconf import conf
import autolens as al
import autolens.plot as aplt
import autofit as af

# %%
"""
Set up the workspace, config and output paths.
"""

# %%
workspace_path = "/path/to/user/autolens_workspace"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
"""
As per usual, load the _Imaging_ data, create the _Mask_ and plot them. In this strong lensing dataset:

    - The lens galaxy's _LightProfile_ is omitted_.
    - The lens galaxy's _MassProfile_ is an _EllipticalIsothermal_.
    - The source galaxy's _LightProfile_ is an _EllipticalExponential_.

"""

# %%
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"
dataset_path = f"{workspace_path}/dataset/{dataset_label}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.1,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Model__

We compose our lens model using _GalaxyModel_ objects, which represent the galaxies we fit to our data. In this 
example our lens mooel is:

    - An _EllipticalIsothermal_ _MassProfile_ for the lens galaxy's mass (5 parameters).
    - An _EllipticalSersic_ _LightProfile_ for the source galaxy's light (6 parameters).

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""

# %%
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

# %%
"""
__Settings__

[checkout 'autolens_workspace/examples/model/customize/settings.py' for a description of all phase settings
searches in PyAutoLens if you haven't already.]

Next, we specify the *PhaseSettingsImaging*, which describe how the model is fitted to the data in the log likelihood
function. Given phase 1 is an initialization phase where we prioritize fast run times over accuracy, we choose the 
following setting:

    - We use a regular *Grid* to fit the data (we did this in the 'beginner' tutorials, but as discussed in
     'autolens_workspace/examples/grids.py' this grid may suffer inaccuracies. We will use a more accurate but 
     computationally expensive grid in phase 2.      
    
    - The grid has a sub-size of 1, providing the fastest but least accurate evaluation of the mass profile's 
      deflection angles and source galaxy's light.
"""

# %%
settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=2)

# %%
"""
__Search__

[checkout 'autolens_workspace/examples/model/customize/non_linear_searches.py' for a description of non-linear 
searches in PyAutoLens if you haven't already.]

In phase 1, we use a fast _NonLinearSearch_ technique called a 'optimizer', which only seeks to maximize the log 
likelihood of the fit (and does not waste time determining the errors on parameters). We'll use the 'particle swarm 
optimizer algorithm *PySwarms* (https://pyswarms.readthedocs.io/en/latest/index.html) using:

    - 30 particles to sample parameter space.
    - 100 iterations per particle, giving a total of 3000 iterations.
    
Performing the model-fit in 3000 iterations is significantly faster than the _Dynesty_ fits perforomed in other 
example scripts, that often require > 20000 - 50000 iterations.
"""

# %%
search = af.PySwarmsGlobal(n_particles=10, iters=5)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

    '/autolens_workspace/output/examples/linking/lens_sie__source_sersic/phase_1'.
"""

# %%
phase_1 = al.PhaseImaging(
    phase_name="phase_1",
    folders=["examples", "linking", dataset_name],
    settings=settings,
    galaxies=dict(lens=lens, source=source),
    search=search,
)

phase_1_result = phase_1.run(dataset=imaging, mask=mask)

# %%
"""
Before reading on to phase 2, you may wish to inspect the results of the phase 1 model-fit to ensure the fast
non-linear search has provided a reasonable accurate lens model.
"""

# %%
"""
__Model Linking__

We use the results of phase 1 to create the _GalaxyModel_ components that we fit in phase 2.

The term 'model' below tells PyAutoLens to pass the lens and source models as model-components that are to be fitted
for by the non-linear search. In other linking examples, we'll see other ways to pass prior results.
"""

# %%
lens = phase_1_result.model.galaxies.lens
source = phase_1_result.model.galaxies.source

# %%
"""
__Settings__

[checkout 'autolens_workspace/examples/model/customize/settings.py' for a description of all phase settings
searches in PyAutoLens if you haven't already.]

Next, we specify the *PhaseSettingsImaging*, which describe how the model is fitted to the data in the log likelihood
function. Given phase 2 is the final phase where we want an accurate model and precise errors we choose the 
following setting:

    - We use a *GridIterate* to fit the data, which is more computationally expensive than a regular *Grid* but 
      iterative refines the source-light profile to a threshold accuracy ensuring a more accurate fit. 

    - The grid has a fractional accurate of 0.9999, ensuring the source light profile is evaluated to 99.99% of 
      the correct value.
"""

# %%
settings = al.PhaseSettingsImaging(
    grid_class=al.GridIterate, fractional_accuracy=0.9999
)

# %%
"""
__Search__

[checkout 'autolens_workspace/examples/model/customize/non_linear_searches.py' for a description of non-linear 
searches in PyAutoLens if you haven't already.]

In phase 2, we use the nested sampling algorithm _Dynesty_, which we used in the beginner examples. _Dynesty_ fully
maps out the posterior in parameter space, taking longer to run but providing errors on every model parameter. 

We can use fewer live points and a faster sampling efficiency than we did in the beginner tutorials, given that
prior linking now informs _Dynesty_ where to search parameter space.
"""

# %%
search = af.DynestyStatic(n_live_points=30, facc=0.8)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

    '/autolens_workspace/output/examples/linking/lens_sie__source_sersic/phase_2'.

You may want to checkout the 'model.info' file in this folder, to see how the priors are now GaussianPrior's
with means centred on the results of phase 1.    
"""

# %%
phase_2 = al.PhaseImaging(
    phase_name="phase_2",
    folders=["examples", "linking", dataset_name],
    settings=settings,
    galaxies=dict(lens=lens, source=source),
    search=search,
)

phase_2.run(dataset=imaging, mask=mask)
