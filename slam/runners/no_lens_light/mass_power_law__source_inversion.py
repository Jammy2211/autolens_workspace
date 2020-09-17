# %%
"""
__SLaM (Source, Light and Mass)__

This SLaM pipeline runner loads a strong lens dataset and analyses it using a SLaM lens modeling pipeline.

__THIS RUNNER__

Using two source pipelines and a mass pipeline this runner fits _Imaging_ of a strong lens system, where in the final
phase of the pipeline:

 - The lens galaxy's _LightProfile_ is modeled as an _EllipticalSersic_.
 - The lens galaxy's _MassProfile_ is modeled as an _EllipticalPowerLaw_.
 - The source galaxy is modeled using an _Inversion_.

This uses the SLaM pipelines:

 'slam/no_lens_light/source__mass_sie__source_parametric.py'.
 'slam/no_lens_light/source__mass_sie__source_inversion.py'.
 'slam/no_lens_light/mass__mass_power_law__source.py'.

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
dataset_label = "no_lens_light"
dataset_name = "mass_sie__source_sersic"
pixel_scales = 0.1

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/no_lens_light/mass_sie__source_sersic'
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

# %%
"""
_Inversion_'s may infer unphysical solution where the source reconstruction is a demagnified reconstruction of the 
lensed source (see **HowToLens** chapter 4). 

To prevent this, auto-positioning is used, which uses the lens mass model of earlier phases to automatically set 
positions and a threshold that resample inaccurate mass models (see 'examples/model/positions.py').

The *auto_positions_factor* is a factor that the threshold of the inferred positions using the previous mass model are 
multiplied by to set the threshold in the next phase. The *auto_positions_minimum_threshold* is the minimum value this
threshold can go to, even after multiplication.
"""

# %%
settings_lens = al.SettingsLens(
    auto_positions_factor=3.0, auto_positions_minimum_threshold=0.8
)

settings = al.SettingsPhaseImaging(
    settings_masked_imaging=settings_masked_imaging, settings_lens=settings_lens
)


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

The _SetupHyper_ determines which hyper-mode features are used during the model-fit as is used identically to the
hyper pipeline examples.

The _SetupHyper_ object has a new input available, 'hyper_fixed_after_source', which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and _SLaMPipelineMass_ pipelines, model comparison can be performed in a consistent fashion.
"""

# %%
hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=False,
    hyper_background_noise=False,
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

setup_mass = al.SetupMassTotal(
    mass_profile=al.mp.EllipticalIsothermal, no_shear=False, mass_centre=(0.0, 0.0)
)
setup_source = al.SetupSourceSersic()

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_source=setup_source
)

# %%
"""
__SLaMPipelineSourceInversion__

The Source inversion pipeline aims to initialize a robust model for the source galaxy using an _Inversion_.

_SLaMPipelineSourceInversion_ determines the _Inversion_ used by the inversion source pipeline. A full description of all 
options can be found ? and ?.

By default, this again assumes _EllipticalIsothermal_ profile for the lens galaxy's mass model.

For this runner the _SLaMPipelineSourceInversion_ customizes:

 - The _Pixelization_ used by the _Inversion_ of this pipeline.
 - The _Regularization_ scheme used by of this pipeline.

The _SLaMPipelineSourceInversion_ use's the _SetupMass_ of the _SLaMPipelineSourceParametric_.

The _SLaMPipelineSourceInversion_ determines the source model used in the _SLaMPipelineLight_ and _SLaMPipelineMass_ pipelines, which in this
example therefore both use an _Inversion_.
"""

setup_source = al.SetupSourceInversion(
    pixelization=al.pix.VoronoiBrightnessImage, regularization=al.reg.AdaptiveBrightness
)

pipeline_source_inversion = al.SLaMPipelineSourceInversion(setup_source=setup_source)

# %%
"""
__SLaMPipelineMass__

The _SLaMPipelineMass_ pipeline fits the model for the lens galaxy's total mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens galaxy's mass is input into _SLaMPipelineMassTotal_ and this runner uses the 
default of an _EllipticalPowerLaw_ in this example.

For this runner the _SLaMPipelineMass_ customizes:

 - The _MassProfile_ fitted by the pipeline.
 - If there is an _ExternalShear_ in the mass model or not.
"""

setup_mass = al.SetupMassTotal(mass_profile=al.mp.EllipticalPowerLaw, no_shear=False)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

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
    pipeline_source_inversion=pipeline_source_inversion,
    pipeline_mass=pipeline_mass,
)

# %%
"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then add the pipelines together and run this summed pipeline, which runs each individual pipeline back-to-back.
"""

# %%
from autolens_workspace.slam.pipelines.no_lens_light import source__sersic
from autolens_workspace.slam.pipelines.no_lens_light import source__inversion
from autolens_workspace.slam.pipelines.no_lens_light import mass__total

source__sersic = source__sersic.make_pipeline(slam=slam, settings=settings)
source__inversion = source__inversion.make_pipeline(slam=slam, settings=settings)
mass__total = mass__total.make_pipeline(slam=slam, settings=settings)

pipeline = source__sersic + source__inversion + mass__total

pipeline.run(dataset=imaging, mask=mask)
