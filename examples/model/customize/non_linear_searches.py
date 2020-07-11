# %%

"""
__Example: Non-linear Searches__

In the 'beginner' examples all model-fits were performed using the nested sampling algorithm _Dynesty_, which is a
very effective non-linear search algorithm for lens modeling, but may not always be the optimal choice for your
problem. In this example we will fit strong lens data using a variety of non-linear searches.
"""

# %%
"""
In this example script, we will fit imaging of a strong lens system where:

 - The lens galaxy's _LightProfile_ is omitted (and is not present in the simulated data.
 - The lens galaxy's _MassProfile_ is fitted with an _EllipticalIsothermal_.
 - The source galaxy's _LightProfile_ is fitted with an _EllipticalSersic_.

"""

# %%
"""Setup the path to the autolens workspace, using the project pyprojroot which determines it automatically."""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# %%
"""Set up the config and output paths."""

# %%
from autoconf import conf

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
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "lens_sie__source_sersic"
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

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
function. Given we want a fast-run time to test each non-linear search quickly, we choose the following setting:

 - We use a regular *Grid* to fit the data, which evaluates the lens deflection angles and source light quickly
      at the expense of accuracy. 
"""

# %%
settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=2)

# %%
"""
__Searches__

Below we use the following non-linear searches:

    1) Nested Sampler.
    2) Optimize.
    3) MCMC
"""

# %%
"""
__Nested Sampling__

To begin, lets again use the nested sampling method _Dynesty_ that we have used in all examples up to now. We've seen 
that the method is very effective, always locating a solution that fits the lens data well.
"""

# %%
search = af.DynestyStatic(n_live_points=50)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/customize/lens_sie__source_sersic/phase__nested_sampling/
    settings__grid_sub_2/dynesty__'.
"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__non_linear_searches",
    folders=["examples", "customize", dataset_name],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

result = phase.run(dataset=imaging, mask=mask)

# %%
"""
__Optimizer__

Now, lets use a fast _NonLinearSearch_ technique called an 'optimizer', which only seeks to maximize the log 
likelihood of the fit and does not attempt to infer the errors on the model parameters. Optimizers are useful when we
want to find a lens model that fits the data well, but do not care about the full posterior of parameter space (e.g.
the errors). 

We'll use the 'particle swarm optimizer algorithm *PySwarms* (https://pyswarms.readthedocs.io/en/latest/index.html) 
using:

 - 30 particles to sample parameter space.
 - 100 iterations per particle, giving a total of 3000 iterations.
    
Performing the model-fit in 3000 iterations is significantly faster than the _Dynesty_ fits perforomed in other 
example scripts, that often require > 20000 - 50000 iterations.
"""

# %%
search = af.PySwarmsLocal(n_particles=50, iters=5000)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/customize/'.
"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__non_linear_searches",
    folders=["examples", "customize", dataset_name],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

result = phase.run(dataset=imaging, mask=mask)

# %%
"""
__MCMC__
"""

# %%
search = af.Emcee(nwalkers=50, nsteps=1000)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/customize/'.
"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__non_linear_searches",
    folders=["examples", "customize", dataset_name],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

result = phase.run(dataset=imaging, mask=mask)
