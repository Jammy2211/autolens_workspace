"""
__SLaM (Source, Light and Mass)__

This SLaM pipeline runner loads a strong lens dataset and analyses it using a SLaM lens modeling pipeline.

__THIS RUNNER__

Using two source pipelines and a mass pipeline this runner fits `Imaging` of a strong lens system, where in the final
phase of the pipeline:

 - The lens `Galaxy`'s light is omitted from the data and model.
 - The lens `Galaxy`'s total mass distribution is modeled as an `EllipticalPowerLaw`.
 - The source galaxy is modeled using an `Inversion`.

This uses the SLaM pipelines:

 `slam/imaging/no_lens_light/pipelines/source__mass_sie__source_parametric.py`.
 `slam/imaging/no_lens_light/pipelines/source__mass_sie__source_inversion.py`.
 `slam/imaging/no_lens_light/pipelines/mass__mass_power_law__source.py`.

Check them out for a detailed description of the analysis!
"""
from os import path
import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__source_sersic"
pixel_scales = 0.1

dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

"""Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
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

"""
`Inversion`'s may infer unphysical solution where the source reconstruction is a demagnified reconstruction of the 
lensed source (see **HowToLens** chapter 4). 

To prevent this, auto-positioning is used, which uses the lens mass model of earlier phases to automatically set 
positions and a threshold that resample inaccurate mass models (see `examples/model/positions.py`).

The `auto_positions_factor` is a factor that the threshold of the inferred positions using the previous mass model are 
multiplied by to set the threshold in the next phase. The *auto_positions_minimum_threshold* is the minimum value this
threshold can go to, even after multiplication.
"""

settings_lens = al.SettingsLens(
    auto_positions_factor=3.0, auto_positions_minimum_threshold=0.8
)

settings = al.SettingsPhaseImaging(
    settings_masked_imaging=settings_masked_imaging, settings_lens=settings_lens
)

"""
__PIPELINE SETUP__

Transdimensional pipelines used the `SetupPipeline` object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the `SetupPipeline` object, customizing the analysis in that pipeline. Each pipeline therefore
has its own `SetupMass` and `SetupSourceParametric` object.

The `Setup` used in earlier pipelines determine the model used in later pipelines. For example, if the `Source` 
pipeline is given a `Pixelization` and `Regularization`, than this `Inversion` will be used in the subsequent 
`SLaMPipelineMass` pipeline.

The `Setup` again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same `SLaMPipelineSource`. they will reuse those results before branching off to fit different models in the 
`SLaMPipelineLightParametric` and / or `SLaMPipelineMass` pipelines. 
"""

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit as is used identically to the
hyper pipeline examples.

The `SetupHyper` object has a new input available, `hyper_fixed_after_source`, which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and `SLaMPipelineMass` pipelines, model comparison can be performed in a consistent fashion.
"""

hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=None,
    hyper_background_noise=None,
)

"""
__SLaMPipelineSourceParametric__

The parametric source pipeline aims to initialize a robust model for the source galaxy using `LightProfile` objects. 

 Source: This parametric source model is used by the SLaM Mass pipeline that follows, and thus sets the complexity of 
 the parametric source model of the overall fit. 
 
 For this runner the `SetupSourceParametric` customizes: 
 
 - That the bulge of the source `Galaxy` is fitted using an `EllipticalSersic`.
 - There is has no over `LightProfile` components (e.g. a disk, envelope)_.
"""

setup_source = al.SetupSourceParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=None,
    envelope_prior_model=None,
)

"""
 Mass: The `MassProfile` used to model the lens `Galaxy`'s mass. This is changed in the SLaM Mass pipeline that follows.
 
 Our experience with lens modeling has shown the `EllipticalIsothermal` profile is the simpliest model to fit that 
 provide a good fit to the majority of strong lenses.
 
 For this runner the `SetupMassProfile` customizes:

 - That the mass of the lens `Galaxy` is fitted using an `EllipticalIsothermal`.
 - That there is not `ExternalShear` in the mass model (this lens was not simulated with shear and we do not include 
 it in the mass model).
 - That the mass profile centre is (0.0, 0.0) (this assumption will be relaxed in the SLaM Mass Pipeline.
"""

setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalIsothermal,
    with_shear=False,
    mass_centre=(0.0, 0.0),
)

"""We combine the `SetupSource` and `SetupMass` above to create the SLaM parametric Source Pipeline."""

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_mass=setup_mass, setup_source=setup_source
)

"""
__SLaMPipelineSourceInversion__

The Source inversion pipeline aims to initialize a robust model for the source galaxy using an `Inversion`.

_SLaMPipelineSourceInversion_ determines the `Inversion` used by the inversion source pipeline. A full description 
of all options can be found ? and ?.

By default, this again assumes `EllipticalIsothermal` profile for the lens `Galaxy`'s mass model.

For this runner the `SLaMPipelineSourceInversion` customizes:

 - The `Pixelization` used by the `Inversion` of this pipeline.
 - The `Regularization` scheme used by the `Inversion` of this pipeline.

The `SLaMPipelineSourceInversion` use`s the `SetupMass` of the `SLaMPipelineSourceParametric`.

The `SLaMPipelineSourceInversion` determines the source model used in the `SLaMPipelineLightParametric` and 
`SLaMPipelineMass` pipelines, which in this example therefore both use an `Inversion`.
"""

setup_source = al.SetupSourceInversion(
    pixelization_prior_model=al.pix.VoronoiBrightnessImage,
    regularization_prior_model=al.reg.AdaptiveBrightness,
)

pipeline_source_inversion = al.SLaMPipelineSourceInversion(setup_source=setup_source)

"""
__SLaMPipelineMass__

The `SLaMPipelineMass` pipeline fits the model for the lens `Galaxy`'s total mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens `Galaxy`'s mass is input into `SLaMPipelineMassTotal` and this runner uses the 
default of an `EllipticalPowerLaw` in this example.

For this runner the `SLaMPipelineMass` customizes:

 - The `MassProfile` fitted by the pipeline.
 - If there is an `ExternalShear` in the mass model or not (this lens was not simulated with shear and 
   we do not include it in the mass model).
"""

setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalPowerLaw, with_shear=False
)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

"""
__SLaM__

We combine all of the above `SLaM` pipelines into a `SLaM` object.

The `SLaM` object contains a number of methods used in the make_pipeline functions which are used to compose the model 
based on the input values. It also handles pipeline tagging and path structure.
"""

slam = al.SLaM(
    path_prefix=path.join("slam", dataset_name),
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_source_inversion=pipeline_source_inversion,
    pipeline_mass=pipeline_mass,
)

"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then add the pipelines together and run this summed pipeline, which runs each individual pipeline back-to-back.
"""

from pipelines import source__parametric
from pipelines import source__inversion
from pipelines import mass__total

source__parametric = source__parametric.make_pipeline(slam=slam, settings=settings)
source_results = source__parametric.run(dataset=imaging, mask=mask)

source__inversion = source__inversion.make_pipeline(
    slam=slam, settings=settings, source_parametric_results=source_results
)
source_results = source__inversion.run(dataset=imaging, mask=mask)

mass__total = mass__total.make_pipeline(
    slam=slam, settings=settings, source_results=source_results
)
mass_results = mass__total.run(dataset=imaging, mask=mask)
