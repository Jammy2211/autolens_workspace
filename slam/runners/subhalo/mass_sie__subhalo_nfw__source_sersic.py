# %%
"""
__SLaM (Source, Light and Mass)__

This SLaM pipeline runner loads a strong lens dataset and analyses it using a SLaM lens modeling
pipeline.

__THIS RUNNER__

Using 1 source pipeline, a mass pipeline and a subhalo pipeline this runner fits _Imaging_ of a strong lens system,
where in the final phase of the pipeline:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's _MassProfile_ is modeled as an _EllipticalIsothermal_.
 - A dark matter subhalo's within the lens galaxy is modeled as a _SphericalNFWMCRLudLow_.
 - The source galaxy is modeled as an _EllipticalSersic_.

This runner uses the SLaM pipelines:

 'slam/no_lens_light/source__mass_sie__source_parametric.py'.
 'slam/no_lens_light/mass__mass_power_law__source.py'.
 'slam/no_lens_light/subhalo__mass__subhalo_nfw__source.py'.

Check them out for a detailed description of the analysis!
"""

# %%
"""Use the WORKSPACE environment variable to determine the path to the autolens workspace."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
"""Use this path to explicitly set the config path and output path."""

# %%
from autoconf import conf

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "subhalo"
dataset_name = "mass_sie__subhalo_nfw__source_sersic"
pixel_scales = 0.05

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/no_lens_light/mass_sie__subhalo_nfw__source_sersic'
"""

# %%
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

# %%
"""Using the dataset path, load the data (image, noise-map, PSF) as an _Imaging_ object from .fits files."""

# %%
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Settings__

The _SettingsPhaseImaging_ describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the 'autolens_workspace/examples/model' example scripts, with a 
complete description of all settings given in 'autolens_workspace/examples/model/customize/settings.py'.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)


# %%
"""
__PIPELINE SETUP__

Transdimensional pipelines used the _SetupPipeline_ object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the _SetupPipeline_ object, customizing the analysis in that pipeline. Each pipeline therefore
has its own _SetupMass_, _SetupLight_ and _SetupSource_ object.

The _Setup_ used in earlier pipelines determine the model used in later pipelines. For example, if the _Source_ 
pipeline is given a _Pixelization_ and _Regularization_, than this _Inversion_ will be used in the subsequent _SLaMPipelineLight_ and 
Mass pipelines. The assumptions regarding the lens light chosen by the _Light_ object are carried forward to the 
_Mass_  pipeline.

The _Setup_ again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same _SLaMPipelineSource_) they will reuse those results before branching off to fit different models in the _SLaMPipelineLight_ 
and / or _SLaMPipelineMass_ pipelines. 
"""

# %%
"""
__HYPER SETUP__

The _SetupHyper_ determines which hyper-mode features are used during the model-fit and is used identically to the
hyper pipeline examples.

The _SetupHyper_ object has a new input available, 'hyper_fixed_after_source', which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and _SLaMPipelineMass_ pipelines, model comparison can be performed in a consistent fashion.
"""

# %%
hyper = al.SetupHyper(
    hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
)

# %%
"""
__SLaMPipelineSourceParametric__

The parametric source pipeline aims to initialize a robust model for the source galaxy using _LightProfile_ objects. 

_SLaMPipelineSourceParametric_ determines the source model used by the parametric source pipeline. A full description of all 
options can be found ? and ?.

By default, this assumes an _EllipticalIsothermal_ profile for the lens galaxy's mass. Our experience with lens 
modeling has shown they are the simpliest models that provide a good fit to the majority of strong lenses.

For this runner the _SLaMPipelineSourceParametric_ customizes:

 - The _MassProfile_ fitted by the pipeline (and the following _SLaMPipelineSourceInversion_.
 - If there is an _ExternalShear_ in the mass model or not.
"""

setup_mass = al.SetupMassTotal(mass_profile=al.mp.EllipticalIsothermal, no_shear=False)
setup_source = al.SetupSourceSersic()

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_source=setup_source
)

# %%
"""
__SLaMPipelineMassTotal__

The _SLaMPipelineMassTotal_ pipeline fits the model for the lens galaxy's total mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens galaxy's mass is input into _SLaMPipelineMass_ and this runner uses an 
_EllipticalIsothermal_ in this example.

For this runner the _SLaMPipelineMass_ customizes:

 - The _MassProfile_ fitted by the pipeline.
 - If there is an _ExternalShear_ in the mass model or not.
"""

setup_mass = al.SetupMassTotal(mass_profile=al.mp.EllipticalIsothermal, no_shear=False)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

# %%
"""
__SetupSubhalo__

The final pipeline fits the lens and source model including a _SphericalNFW_ subhalo, using a grid-search of non-linear
searchesn. 

A full description of all options can be found ? and ?.

The models used to represent the lens galaxy's mass and the source are those used in the previous pipelines.

For this runner the _SetupSubhalo_ customizes:

 - If the lens galaxy mass is treated as a model (all free parameters) or instance (all fixed) during the subhalo 
   detection grid search.
 - If the source galaxy (parametric or _Inversion) is treated as a model (all free parameters) or instance (all fixed) 
   during the subhalo detection grid search.
 - The NxN size of the grid-search.
"""
setup_subhalo = al.SetupSubhalo(mass_is_model=True, source_is_model=True, grid_size=5)

# %%
"""
__SLaM__

We combine all of the above _SLaM_ pipelines into a _SLaM_ object.

The _SLaM_ object contains a number of methods used in the make_pipeline functions which are used to compose the model 
based on the input values. It also handles pipeline tagging and path structure.
"""

slam = al.SLaM(
    folders=["slam", f"{dataset_type}_{dataset_label}", dataset_name],
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_mass=pipeline_mass,
    setup_subhalo=setup_subhalo,
)

# %%
"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then add the pipelines together and run this summed pipeline, which runs each individual pipeline back-to-back.
"""

# %%
from autolens_workspace.slam.pipelines.no_lens_light import source__sersic
from autolens_workspace.slam.pipelines.no_lens_light import mass__total
from autolens_workspace.slam.pipelines.no_lens_light import subhalo

source__sersic = source__sersic.make_pipeline(slam=slam, settings=settings)
mass__total = mass__total.make_pipeline(slam=slam, settings=settings)
subhalo = subhalo.make_pipeline(slam=slam, settings=settings)

pipeline = source__sersic + mass__total + subhalo

pipeline.run(dataset=imaging, mask=mask)
