# %%
"""
__SLaM (Source, Light and Mass)__

This SLaM pipeline runner loads a strong lens dataset and analyses it using a SLaM lens modeling pipeline.

__THIS RUNNER__

Using two source pipelines, a light pipeline and a mass pipeline this runner fits `Imaging` of a strong lens system
where in the final phase of the pipeline:

 - The lens galaxy`s `LightProfile``s are modeled as an `EllipticalSersic` + `EllipticalExponential`, representing
   a bulge + disk model.
 - The lens galaxy`s stellar `MassProfile` is fitted using the `EllipticalSersic` + `EllipticalExponential` of the
    `LightProfile`, where it is converted to a stellar mass distribution via constant mass-to-light ratios.
 - The lens galaxy`s dark matter `MassProfile` is modeled as a `SphericalNFW`.
 - The source galaxy`s `LightProfile` is modeled using an `Inversion`.

This runner uses the SLaM pipelines:

 `slam/with_lens_light/source__sersic.py`.
 `slam/with_lens_light/source___inversion.py`.
 `slam/with_lens_light/light__bulge_disk.py`.
 `slam/with_lens_light/mass__light_dark__bulge_disk.py`.

Check them out for a detailed description of the analysis!
"""

# %%
"""Use the WORKSPACE environment variable to determine the path to the `autolens_workspace`."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
""" AUTOLENS + DATA SETUP """
import autolens as al
import autolens.plot as aplt

# %%
"""Specify the dataset type, label and name, which we use to determine the path we load the data from."""
dataset_type = "imaging"
dataset_label = "light_dark"
dataset_name = "light_bulge_disk__mass_mlr_nfw__source_sersic"
pixel_scales = 0.1

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
`/autolens_workspace/dataset/imaging/light_bulge_disk__mass_mlr_nfw__source_parametric`
"""

# %%
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

# %%
"""Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""

# %%
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

# %%
"""
__Settings__

The `SettingsPhaseImaging` describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the `autolens_workspace/examples/model` example scripts, with a 
complete description of all settings given in `autolens_workspace/examples/model/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
"""
Due to the slow deflection angle calculation of the `EllipticalSersic` and `EllipticalExponential` `MassProfile``.
we use `GridInterpolate` objects to speed up the analysis. This is specified separately for the `Grid` used to fit
the source `LightProfile` and perform the `Inversion`.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(
    grid_class=al.GridInterpolate,
    grid_inversion_class=al.GridInterpolate,
    pixel_scales_interp=0.1,
)

# %%
"""
_Inversion_`s may infer unphysical solution where the source reconstruction is a demagnified reconstruction of the 
lensed source (see **HowToLens** chapter 4). 

To prevent this, auto-positioning is used, which uses the lens mass model of earlier phases to automatically set 
positions and a threshold that resample inaccurate mass models (see `examples/model/positions.py`).

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

Transdimensional pipelines used the `SetupPipeline` object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specific aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the `SetupPipeline` object, customizing the analysis in that pipeline. Each pipeline therefore
has its own `SetupMass`, `SetupLight` and `SetupSource` object.

The `Setup` used in earlier pipelines determine the model used in later pipelines. For example, if the `Source` 
pipeline is given a `Pixelization` and `Regularization`, than this `Inversion` will be used in the subsequent 
_SLaMPipelineLight_ and Mass pipelines. The assumptions regarding the lens light chosen by the `Light` object are 
carried forward to the `Mass`  pipeline.

The `Setup` again tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same `SLaMPipelineSource`. they will reuse those results before branching off to fit different models in the 
_SLaMPipelineLight_ and / or `SLaMPipelineMass` pipelines. 
"""

# %%
"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit and is used identically to the
hyper pipeline examples.

The `SetupHyper` object has a new input available, `hyper_fixed_after_source`, which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
_SLaMPipelineLight_ and `SLaMPipelineMass` pipelines, model comparison can be performed in a consistent fashion.
"""

# %%
hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_image_sky=False,
    hyper_background_noise=False,
    hyper_fixed_after_source=True,
)

# %%
"""
__SLaMPipelineSourceParametric__

The parametric source pipeline aims to initialize a robust model for the source galaxy using `LightProfile` objects. 

_SLaMPipelineSourceParametric_ determines the source model used by the parametric source pipeline. A full description of all 
options can be found ? and ?.

By default, this assumes an `EllipticalIsothermal` profile for the lens galaxy`s mass and an `EllipticalSersic` + 
_EllipticalExponential_ model for the lens galaxy`s light. Our experience with lens modeling has shown they are the 
simplest models that provide a good fit to the majority of strong lenses.

For this runner the `SLaMPipelineSourceParametric` customizes:

 - The `MassProfile` fitted by the pipeline (and the following `SLaMPipelineSourceInversion`..
 - If there is an `ExternalShear` in the mass model or not.
"""

setup_light = al.SetupLightBulgeDisk()
setup_mass = al.SetupMassTotal(mass_profile=al.mp.EllipticalIsothermal, no_shear=False)
setup_source = al.SetupSourceSersic()

pipeline_source_parametric = al.SLaMPipelineSourceParametric(
    setup_light=setup_light, setup_mass=setup_mass, setup_source=setup_source
)

# %%
"""
__SLaMPipelineSourceInversion__

The Source inversion pipeline aims to initialize a robust model for the source galaxy using an `Inversion`.

_SLaMPipelineSourceInversion_ determines the `Inversion` used by the inversion source pipeline. A full description of all 
options can be found ? and ?.

By default, this again assumes `EllipticalIsothermal` profile for the lens galaxy`s mass and an `EllipticalSersic` + 
_EllipticalExponential_ model for the lens galaxy`s light.

For this runner the `SLaMPipelineSourceInversion` customizes:

 - The `Pixelization` used by the `Inversion` of this pipeline.
 - The `Regularization` scheme used by of this pipeline.
 - If a fixed number of pixels are used by the `Inversion`.

The `SLaMPipelineSourceInversion` use`s the `SetupLight` and `SetupMass` of the `SLaMPipelineSourceParametric`.

The `SLaMPipelineSourceInversion` determines the source model used in the `SLaMPipelineLight` and `SLaMPipelineMass` pipelines, which in this
example therefore both use an `Inversion`.
"""

setup_source = al.SetupSourceInversion(
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
    inversion_pixels_fixed=1200,
)

pipeline_source_inversion = al.SLaMPipelineSourceInversion(setup_source=setup_source)

# %%
"""
__SLaMPipelineLight__

The `SLaMPipelineLight` pipeline fits the model for the lens galaxy`s bulge + disk light model. 

A full description of all options can be found ? and ?.

 The model used to represent the lens galaxy`s light is input into `SLaMPipelineLight` below and this runner uses an 
 `EllipticalSersic` + `EllipticalExponential` bulge-disk model in this example.
 
For this runner the `SLaMPipelineLight` customizes:

 - The alignment of the centre and elliptical components of the bulge and disk.
 - If the disk is modeled as an `EllipticalExponential` or `EllipticalSersic`.

The `SLaMPipelineLight` uses the mass model fitted in the previous `SLaMPipelineSource``..

The `SLaMPipelineLight` and imported light pipelines determine the lens light model used in `Mass` pipelines.
"""

# %%
setup_light = al.SetupLightBulgeDisk(
    align_bulge_disk_centre=True,
    align_bulge_disk_elliptical_comps=False,
    disk_as_sersic=False,
)

pipeline_light = al.SLaMPipelineLight(setup_light=setup_light)

# %%
"""
__SLaMPipelineMass__

The `SLaMPipelineMass` pipeline fits the model for the lens galaxy`s decomposed stellar and dark matter mass distribution. 

A full description of all options can be found ? and ?.

The model used to represent the lens galaxy`s mass is an `EllipticalSersic` and `EllipticalExponential` 
_LightMassProfile_ representing the bulge and disk fitted in the previous pipeline, alongside a `SphericalNFW` for the
dark matter halo.

For this runner the `SLaMPipelineMass` customizes:

 - If there is an `ExternalShear` in the mass model or not.
"""

setup_mass = al.SetupMassLightDark(no_shear=False)

pipeline_mass = al.SLaMPipelineMass(setup_mass=setup_mass)

# %%
"""
__SLaM__

We combine all of the above `SLaM` pipelines into a `SLaM` object.

The `SLaM` object contains a number of methods used in the make_pipeline functions which are used to compose the model 
based on the input values. It also handles pipeline tagging and path structure.
"""

slam = al.SLaM(
    folders=["slam", f"{dataset_type}_{dataset_label}", dataset_name],
    setup_hyper=hyper,
    pipeline_source_parametric=pipeline_source_parametric,
    pipeline_source_inversion=pipeline_source_inversion,
    pipeline_light=pipeline_light,
    pipeline_mass=pipeline_mass,
)

# %%
"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

We then add the pipelines together and run this summed pipeline, which runs each individual pipeline back-to-back.
"""

# %%
from autolens_workspace.slam.pipelines.with_lens_light import source__sersic
from autolens_workspace.slam.pipelines.with_lens_light import source__inversion
from autolens_workspace.slam.pipelines.with_lens_light import light__bulge_disk
from autolens_workspace.slam.pipelines.with_lens_light import (
    mass__light_dark__bulge_disk,
)

source__sersic = source__sersic.make_pipeline(slam=slam, settings=settings)
source__inversion = source__inversion.make_pipeline(slam=slam, settings=settings)
light__bulge_disk = light__bulge_disk.make_pipeline(slam=slam, settings=settings)
mass__light_dark__bulge_disk = mass__light_dark__bulge_disk.make_pipeline(
    slam=slam, settings=settings
)

pipeline = (
    source__sersic
    + source__inversion
    + light__bulge_disk
    + mass__light_dark__bulge_disk
)

pipeline.run(dataset=imaging, mask=mask)
