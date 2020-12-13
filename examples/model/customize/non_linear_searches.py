"""
__Example: Non-linear Searches__

In the `beginner` examples all model-fits were performed using the nested sampling algorithm `Dynesty`, which is a
very effective `NonLinearSearch` for lens modeling, but may not always be the optimal choice for your
problem. In this example we fit strong lens data using a variety of non-linear searches.
"""

"""
In this example script, we fit `Imaging` of a strong lens system where:

 - The lens `Galaxy`'s light is omitted (and is not present in the simulated data).
 - The lens `Galaxy`'s total mass distribution is modeled as an `EllipticalIsothermal`.
 - The source `Galaxy`'s light is modeled parametrically as an `EllipticalSersic`.

"""

"""
As per usual, load the `Imaging` data, create the `Mask2D` and plot them. In this strong lensing dataset:

 - The lens `Galaxy`'s light is omitted_.
 - The lens `Galaxy`'s total mass distribution is an `EllipticalIsothermal`.
 - The source `Galaxy`'s `LightProfile` is an `EllipticalExponential`.

"""

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

"""
__Model__

We compose our lens model using `GalaxyModel` objects, which represent the galaxies we fit to our data. In this 
example our lens mooel is:

 - An `EllipticalIsothermal` `MassProfile`.for the lens `Galaxy`'s mass (5 parameters).
 - An `EllipticalSersic` `LightProfile`.for the source `Galaxy`'s light (6 parameters).

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""

lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic)

"""
__Settings__

Next, we specify the `SettingsPhaseImaging`, which in this example simmply use the default values used in the beginner
examples.
"""

settings_masked_imaging = al.SettingsMaskedImaging()

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__Searches__

Below we use the following non-linear searches:

    1) Nested Sampler.
    2) Optimize.
    3) MCMC
"""

"""
__Nested Sampling__

To begin, lets again use the nested sampling method `Dynesty` that we have used in all examples up to now. We've seen 
that the method is very effective, always locating a solution that fits the lens data well.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/customize/mass_sie__source_sersic/phase__nested_sampling/
    settings__grid_sub_2/dynesty__`.
"""

search = af.DynestyStatic(
    path_prefix=path.join("examples", "customize", dataset_name),
    name="phase_non_linear_searches",
    n_live_points=50,
)

"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.
"""

phase = al.PhaseImaging(
    search=search,
    galaxies=af.CollectionPriorModel(lens=lens, source=source),
    settings=settings,
)

result = phase.run(dataset=imaging, mask=mask)

"""
__Optimizer__

Now, lets use a fast `NonLinearSearch` technique called an `optimizer`, which only seeks to maximize the log 
likelihood of the fit and does not attempt to infer the errors on the model parameters. Optimizers are useful when we
want to find a lens model that fits the data well, but do not care about the full posterior of parameter space (e.g.
the errors). 

we'll use the `particle swarm optimizer algorithm *PySwarms* (https://pyswarms.readthedocs.io/en/latest/index.html) 
using:

 - 30 particles to sample parameter space.
 - 100 iterations per particle, giving a total of 3000 iterations.
    
Performing the model-fit in 3000 iterations is significantly faster than the `Dynesty` fits perforomed in other 
example scripts, that often require > 20000 - 50000 iterations.
"""

search = af.PySwarmsLocal(
    path_prefix=f"examples/customize/{dataset_name}",
    name="phase__non_linear_searches",
    n_particles=50,
    iters=5000,
)

"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/customize`.
"""

phase = al.PhaseImaging(
    search=search,
    galaxies=af.CollectionPriorModel(lens=lens, source=source),
    settings=settings,
)

result = phase.run(dataset=imaging, mask=mask)

"""
__MCMC__
"""

search = af.Emcee(
    path_prefix=f"examples/customize/{dataset_name}",
    name="phase_non_linear_searches",
    nwalkers=50,
    nsteps=1000,
)

"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/customize`.
"""

phase = al.PhaseImaging(
    search=search,
    galaxies=af.CollectionPriorModel(lens=lens, source=source),
    settings=settings,
)

result = phase.run(dataset=imaging, mask=mask)
