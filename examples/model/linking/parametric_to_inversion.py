# %%
"""
__Example: Linking Parametric To Inversion__

Before reading this example, make sure you have read the 'autolens_workspace/examples/model/linking/api.py'
example script, which describes phase linking and details the API for this.
"""

# %%
"""
In this example, we link two phases, where:

 - Both phases model the lens galaxy's mass as an _EllipticalIsothermal_, with the lens's light omitted.
    
 - The first phase models the source galaxy using a parametric _EllipticalSersic_ profile.
    
 - The second phase models the source galaxy using an _Inversion_, where its _EllipticalIsothermal_ mass model
      priors are initialized using the results of phase 1.

There are a number of benefits to linking a parametric source model to an _Inversion, as opposed to fitting the
_Inversion_ in one phase:

 - Parametric sources are computationally faster to evaluate and fit to the data than an _Inversion_. Thus, although
      the _EllipticalSersic_ carries with it more parameters that the non-linear search will have to fit for, the
      model-fit will be faster overall given the increased speed of each log likelihood evaluation.

 - _Inversion_'s often go to unphysical solutions where the mass model goes to extremely high / low normalizations
      and the source is reconstructed as a demagnified version of the lensed source (see Chapter 4, tutorial 6 for a
      complete description of this effect). A powerful way to prevent this from happening is to initialize the mass
      model with a fit using a parametric source (which does not suffer these unphysical solutions) and use this result
      to ensure the non-linear search samples only the maximal likelihood regions of parameter space.
      
 - To further remove these solutions, we use the 'auto_positions' feature of the _SettingsPhaseImaging_, which use
      the maximum log likelihood mass model of the first phase to determine the positions in the image-plane the
      brightest regions of the lensed source trace too. In phase 2, mass models must trace these positions into a 
      threshold arc-secoond value of one another in the source-plane, ensuring the incorrect solutions corresponding to  
      unphysically large / small mass models are removed.
"""

# %%
"""Setup the path to the autolens workspace, using pyprojroot to determine it automatically."""

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
dataset_name = "mass_sie__source_sersic"
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
 - An _EllipticalSersic_ _LightProfile_ for the source galaxy's light (6 parameters) in phase 1.
 - An _Inversion_ in phase 2 (3 parameters).

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""

# %%
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, sersic=al.lp.EllipticalSersic)

# %%
"""
__Settings__

You should be familiar with the _SettingsPhaseImaging_ object from other example scripts, if not checkout the beginner
examples and 'autolens_workspace/examples/model/customize/settings.py'

In this example we use the 'auto_positions' inputs. These positions correspond to the brightest pixels of the lensed 
source's multiple images When a phase uses positions, during model-fitting they must trace within a threshold value of 
one another for every mass model sampled by the non-linear search. If they do not, the model is discard and resampled. 
The setting below lead to the following behaviour for each phase:

 - In phase 1, because no positions are input into the _Imaging_ dataset, positions are not used and the 
      auto_positions settings do nothing.

 - In phase 2, because there are auto_positions settings, the maximum log likelihood model of phase 1 is used too 
      compute the positions of the lensed source galaxy and the threshold within which they trace to one another. This
      threshold is multiplied by the 'auto_positions_factor' to ensure it is not too small (and thus does not remove
      many plausible mass models). If, after this multiplication, the threshold is below the 
  'auto_positions_minimum_threshold', it is rounded up to this minimum value.
"""

# %%
settings_lens = al.SettingsLens(
    auto_positions_factor=3.0, auto_positions_minimum_threshold=0.2
)

settings = al.SettingsPhaseImaging(settings_lens=settings_lens)

# %%
"""
__Search__

You should be familiar with non-linear searches from other example scripts if not checkout the beginner examples
and 'autolens_workspace/examples/model/customize/non_linear_searches.py'.

In this example, we omit the PriorPasser and will instead use the default values used to pass priors in the config 
file 'autolens_workspace/config/non_linear/nest/DynestyStatic.ini'
"""

# %%
search = af.DynestyStatic(n_live_points=50)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/linking/mass_sie__source_sersic/phase_1'.
"""

# %%
phase1 = al.PhaseImaging(
    phase_name="phase_1",
    folders=["examples", "linking", "parametric_to_inversion"],
    settings=settings,
    galaxies=dict(lens=lens, source=source),
    search=search,
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

We use the results of phase 1 to create the _GalaxyModel_ components that we fit in phase 2.

The term 'model' below tells PyAutoLens to pass the lens and source models as model-components that are to be fitted
for by the non-linear search. In other linking examples, we'll see other ways to pass prior results.
"""

# %%
lens = phase1_result.model.galaxies.lens
source = al.GalaxyModel(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification,
    regularization=al.reg.Constant,
)

# %%
"""
__Search__

In phase 2, we use the nested sampling algorithm _Dynesty_ again.
"""

# %%
search = af.DynestyStatic(n_live_points=40)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/linking/mass_sie__source_sersic/phase_2'.

Note how the 'lens' passed to this phase was set up above using the results of phase 1!
"""

# %%
phase2 = al.PhaseImaging(
    phase_name="phase_2",
    folders=["examples", "linking", "parametric_to_inversion"],
    settings=settings,
    galaxies=dict(lens=lens, source=source),
    search=search,
)

phase2.run(dataset=imaging, mask=mask)

# %%
"""
__Wrap Up__

In this example, we passed used prior passing to initialize a lens mass model using a parametric source and pass this
model to a second phase which modeled the source using an _Inversion_. We won in terms of efficiency and ensuring the
_Inversion_ did not go to an unphysical solution.

__Pipelines__

The next level of PyAutoLens uses _Pipelines_, which link together multiple phases to perform very complex lens 
modeling in robust and efficient ways. Pipelines which fit the source as an _Inversion_, for example:

 'autolens_wokspace/pipelines/no_lens_light/lens_sie__source_inversion.py'

Exploit our ability to first model the source using a parametric profile and then switch to an _Inversion_, to ensure 
more efficient and robust model-fits!
"""
