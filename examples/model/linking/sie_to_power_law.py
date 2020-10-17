# %%
"""
__Example: Linking Parametric To Inversion__

Before reading this example, make sure you have read the `autolens_workspace/examples/model/linking/api.py`
example script, which describes phase linking and details the API for this.
"""

# %%
"""
In this example, we link two phases, where:

 - The first phase models the lens `Galaxy`'s mass as an `EllipticalIsothermal` and the source `Galaxy`'s light as an
      `EllipticalSersic`.
      
 - The second phase models the lens `Galaxy`'s mass an an `EllipticalPoerLaw` and the source `Galaxy`'s light as an
      `EllipticalSersic`.

The `EllipticalPower` is a general form of the `EllipticalIsothermal` and it has one addition parameter relative to the
`EllipticalIsothermal`, the `slope`. This controls the internal mass distriibution of the mass profile, whereby:

 - A higher slope concentrates more mass in the central regions of the `MassProfile` relative to the outskirts. 
 - A lower slope shallows the inner mass distribution reducing its density relative to the outskirts. 

By allowing a `MassProfile` to vary its inner distribution, the non-linear parameter space of the lens model becomes 
significantly more complex, creating a notable degeneracy between the mass model`s mass normalization, ellipticity
and slope. This proves challenging to sample in an efficient and robust manner, especially if our initial samples are
not initalized so as to start sampling in the high likelhood regions of parameter space.

We can use prior passing to perform this initialization!  The `EllipticalIsothermal` profile corresponds to an 
`EllipticalPowerLaw` with a slope = 2.0. Thus, we can first fit an `EllipticalIsothermal` model in a non-linear 
parameter space that does not have the strong degeneracy between mass, ellipticity and axis-ratio, which will 
provide an efficient and robust fit. 

Phase 2 can then fit the `EllipticalPowerLaw`, using prior passing to initialize robust models for both the lens 
`Galaxy`'s mass *and* the source `Galaxy`'s light. 
"""

# %%
"""
As per usual, load the `Imaging` data, create the `Mask2D` and plot them. In this strong lensing dataset:

 - The lens `Galaxy`'s light is omitted_.
 - The lens total mass distribution is an `EllipticalPowerLaw`.
 - The source `Galaxy`'s `LightProfile` is an `EllipticalSersic`.

"""

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "mass_power_law__source_sersic"
dataset_path = f"dataset/{dataset_type}/{dataset_label}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Model__

We compose our lens model using `GalaxyModel` objects, which represent the galaxies we fit to our data. In this 
example our lens model is:

 - An `EllipticalIsothermal` `MassProfile`.for the lens `Galaxy`'s mass (5 parameters) in phase 1.
 - An `EllipticalSersic` `LightProfile`.for the source `Galaxy`'s light (7 parameters) in phase 1.
 - An `EllipticalPowerLaw` `MassProfile`.for the lens `Galaxy`'s mass (6 parameters) in phase 2.
 - An `EllipticalSersic` `LightProfile`.for the source `Galaxy`'s light (7 parameters) in phase 2.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=12 and N=13
for phases 1 and 2 respectively..
"""

# %%
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic)

# %%
"""
__Settings__

You should be familiar with the `SettingsPhaseImaging` object from other example scripts, if not checkout the beginner
examples and `autolens_workspace/examples/model/customize/settings.py`
"""

# %%
settings = al.SettingsPhaseImaging()

# %%
"""
__Search__

You should be familiar with non-linear searches from other example scripts if not checkout the beginner examples
and `autolens_workspace/examples/model/customize/non_linear_searches.py`.

In this example, we omit the PriorPasser and will instead use the default values used to pass priors in the config 
file `autolens_workspace/config/non_linear/nest/DynestyStatic.ini`
"""

# %%
search = af.DynestyStatic(
    path_prefix=f"examples/linking/sie_to_power_law", name="phase[1]", n_live_points=50
)

# %%
"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/linking/sie_to_power_law/lens_power_law__source_bulge/phase[1]`.
"""

# %%
phase1 = al.PhaseImaging(
    search=search, settings=settings, galaxies=dict(lens=lens, source=source)
)

phase1_result = phase1.run(dataset=imaging, mask=mask)

# %%
"""
Before reading on to phase 2, you may wish to inspect the results of the phase 1 model-fit to ensure the fast
non-linear search has provided a reasonably accurate lens model.
"""

# %%
"""
__Model Linking__

We use the results of phase 1 to create the `GalaxyModel` components that we fit in phase 2.

The term `model` below tells PyAutoLens to pass the lens and source models as model-components that are to be fitted
for by the non-linear search. In other linking examples, we'll see other ways to pass prior results.

Because we change the `MassProfile` from an `EllipticalIsothermal` to an `EllipticalPowerLaw`, we cannot simply pass the
mass model above. Instead, we pass each individual parameter of the `EllipticalIsothermal` model, leaving the slope
to retain its default `UniformPrior` which has a lower_limit=1.5 and upper_limit=3.0.
"""

# %%

mass = af.PriorModel(al.mp.EllipticalPowerLaw)

mass.centre = phase1_result.model.galaxies.lens.mass.centre
mass.elliptical_comps = phase1_result.model.galaxies.lens.mass.elliptical_comps
mass.einstein_radius = phase1_result.model.galaxies.lens.mass.einstein_radius

lens = al.GalaxyModel(redshift=0.5, mass=mass)

source = al.GalaxyModel(redshift=1.0, bulge=phase1_result.model.galaxies.source.bulge)

# %%
"""
__Search__

In phase 2, we use the nested sampling algorithm `Dynesty` again.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/linking/sie_to_power_law/lens_power_law__source_bulge/phase[2]`.
"""

# %%
search = af.DynestyStatic(
    path_prefix=f"examples/linking/sie_to_power_law", name="phase[2]", n_live_points=50
)

# %%
"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.

Note how the `lens` passed to this phase was set up above using the results of phase 1!
"""

# %%
phase2 = al.PhaseImaging(
    search=search, settings=settings, galaxies=dict(lens=lens, source=source)
)

phase2.run(dataset=imaging, mask=mask)

# %%
"""
__Wrap Up__

In this example, we passed used prior passing to initialize a lens mass model as an `EllipticalIsothermal` and 
passed its priors to then fit the more complex `EllipticalPowerLaw`. model. This removed difficult-to-fit degeneracies
from the non-linear parameter space in phase 1, providing a more robust and efficient model-fit.

__Pipelines__

The next level of PyAutoLens uses `Pipelines`, which link together multiple phases to perform very complex lens 
modeling in robust and efficient ways. Pipelines which fit a power-law, for example:

 `autolens_wokspace/pipelines/no_lens_light/mass_power_law__source_inversion.py`

Exploit our ability to first model the lens`s mass using an `EllipticalIsothermal` and then switch to an 
`EllipticalPowerLaw`, to ensure more efficient and robust model-fits!
"""
