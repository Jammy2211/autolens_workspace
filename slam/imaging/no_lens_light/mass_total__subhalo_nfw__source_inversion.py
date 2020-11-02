"""
__SLaM (Source, Light and Mass)__

This SLaM pipeline runner loads a strong lens dataset and analyses it using a SLaM lens modeling
pipeline.

__THIS RUNNER__

Using 1 source pipeline, a mass pipeline and a subhalo pipeline this runner fits `Imaging` of a strong lens system,
where in the final phase of the pipeline:

 - The lens `Galaxy`'s light is omitted from the data and model.
 - The lens `Galaxy`'s total mass distribution is modeled as an `EllipticalIsothermal`.
 - A dark matter subhalo`s within the lens galaxy is modeled as a `SphericalNFWMCRLudLow`.
 - The source galaxy is modeled as an `EllipticalSersic`.

This runner uses the SLaM pipelines:

 `slam/imaging/no_lens_light/pipelines/source__mass_sie__source_parametric.py`.
 `slam/imaging/no_lens_light/pipelines/source__mass_sie__source_inversion.py`.
 `slam/imaging/no_lens_light/pipelines/mass__mass_power_law__source.py`.
 `slam/imaging/no_lens_light/pipelines/subhalo__mass__subhalo_nfw__source.py`.

Check them out for a detailed description of the analysis!
"""

import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__subhalo_nfw__source_sersic"
pixel_scales = 0.05

dataset_path = f"dataset/imaging/no_lens_light/{dataset_name}"

"""Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

"""
__Settings__

The `SettingsPhaseImaging` describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the `autolens_workspace/examples/model` example scripts, with a 
complete description of all settings given in `autolens_workspace/examples/model/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline.
"""

settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__PIPELINE SETUP__

Transdimensional pipelines used the `SetupPipeline` object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the `SetupPipeline` object, customizing the analysis in that pipeline. Each pipeline therefore
has its own `SetupMass`, `SetupLightParametric` and `SetupSourceParametric` object.

The `Setup` used in earlier pipelines determine the model used in later pipelines. For example, if the `Source` 
pipeline is given a `Pixelization` and `Regularization`, than this `Inversion` will be used in the subsequent `SLaMPipelineLightParametric` and 
Mass pipelines. The assumptions regarding the lens light chosen by the `Light` object are carried forward to the 
`Mass`  pipeline.

The `Setup` again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same `SLaMPipelineSource`. they will reuse those results before branching off to fit different models in the 
`SLaMPipelineLightParametric` and / or `SLaMPipelineMass` pipelines. 
"""

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit and is used identically to the
hyper pipeline examples.

The `SetupHyper` object has a new input available, `hyper_fixed_after_source`, which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and `SLaMPipelineMass` pipelines, model comparison can be performed in a consistent fashion.
"""

hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=False,
    hyper_background_noise=False,
)

"""
__SLaMPipelineSourceParametric__

The parametric source pipeline aims to initialize a robust model for the source galaxy using `LightProfile` objects. 

_SLaMPipelineSourceParametric_ determines the source model used by the parametric source pipeline. A full description of all 
options can be found ? and ?.

By default, this assumes an `EllipticalIsothermal` profile for the lens `Galaxy`'s mass. Our experience with lens 
modeling has shown they are the simpliest models that provide a good fit to the majority of strong lenses.

For this runner the `SLaMPipelineSourceParametric` customizes:

 - The `MassProfile` fitted by the pipeline (and the following `SLaMPipelineSourceInversion`.
 - If there is an `ExternalShear` in the mass model or not.
"""

setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalIsothermal, with_shear=True
)
setup_source = al.SetupSourceParametric()

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_source=setup_source
)

"""
__SLaMPipelineSourceInversion__

The Source inversion pipeline aims to initialize a robust model for the source galaxy using an `Inversion`.

_SLaMPipelineSourceInversion_ determines the `Inversion` used by the inversion source pipeline. A full description of all 
options can be found ? and ?.

By default, this again assumes `EllipticalIsothermal` profile for the lens `Galaxy`'s mass model.

For this runner the `SLaMPipelineSourceInversion` customizes:

 - The `Pixelization` used by the `Inversion` of this pipeline.
 - The `Regularization` scheme used by the `Inversion` of this pipeline.

The `SLaMPipelineSourceInversion` use`s the `SetupMass` of the `SLaMPipelineSourceParametric`.

The `SLaMPipelineSourceInversion` determines the source model used in the `SLaMPipelineLightParametric` and `SLaMPipelineMass` pipelines, which in this
example therefore both use an `Inversion`.
"""

setup_source = al.SetupSourceInversion(
    pixelization_prior_model=al.pix.VoronoiBrightnessImage,
    regularization_prior_model=al.reg.AdaptiveBrightness,
)

pipeline_source_inversion = al.SLaMPipelineSourceInversion(setup_source=setup_source)

"""
__SLaMPipelineMassTotal__

The `SLaMPipelineMassTotal` pipeline fits the model for the lens `Galaxy`'s total mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens `Galaxy`'s mass is input into `SLaMPipelineMass` and this runner uses an 
`EllipticalIsothermal` in this example.

For this runner the `SLaMPipelineMass` customizes:

 - The `MassProfile` fitted by the pipeline.
 - If there is an `ExternalShear` in the mass model or not.
"""

setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalIsothermal, with_shear=True
)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

"""
__SetupSubhalo__

The final pipeline fits the lens and source model including a `SphericalNFW` subhalo, using a grid-search of non-linear
searchesn. 

A full description of all options can be found ? and ?.

The models used to represent the lens `Galaxy`'s mass and the source are those used in the previous pipelines.

For this runner the `SetupSubhalo` customizes:

 - If the lens galaxy mass is treated as a model (all free parameters) or instance (all fixed) during the subhalo 
   detection grid search.
 - If the source galaxy (parametric or _Inversion) is treated as a model (all free parameters) or instance (all fixed) 
   during the subhalo detection grid search.
 - The NxN size of the grid-search.
"""

setup_subhalo = al.SetupSubhalo(mass_is_model=True, source_is_model=True, grid_size=5)

"""
__SLaM__

We combine all of the above `SLaM` pipelines into a `SLaM` object.

The `SLaM` object contains a number of methods used in the make_pipeline functions which are used to compose the model 
based on the input values. It also handles pipeline tagging and path structure.
"""

slam = al.SLaM(
    path_prefix=f"slam/{dataset_name}",
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_source_inversion=pipeline_source_inversion,
    pipeline_mass=pipeline_mass,
    setup_subhalo=setup_subhalo,
)

"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then add the pipelines together and run this summed pipeline, which runs each individual pipeline back-to-back.
"""

from pipelines import source__parametric
from pipelines import source__inversion
from pipelines import mass__total
from pipelines import subhalo

source__parametric = source__parametric.make_pipeline(slam=slam, settings=settings)
source__inversion = source__inversion.make_pipeline(slam=slam, settings=settings)
mass__total = mass__total.make_pipeline(slam=slam, settings=settings)
subhalo = subhalo.make_pipeline(slam=slam, settings=settings)

pipeline = source__parametric + source__inversion + mass__total + subhalo

pipeline.run(dataset=imaging, mask=mask)
