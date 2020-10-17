# %%
"""
__WELCOME__ 

This transdimensional pipeline runner loads a strong lens dataset and analyses it using a transdimensional lens
modeling pipeline.

Using a pipeline composed of five phases this runner fits `Imaging` of a strong lens system, where in the final phase
of the pipeline:

 - The lens `Galaxy`'s light is modeled parametrically as an `EllipticalChameleon`.
 - The lens `Galaxy`'s light matter mass distribution is fitted with the `EllipticalChameleon` of the
      `LightProfile`, where it is converted to a stellar mass distribution via a constant mass-to-light ratio.
 - The lens `Galaxy`'s dark matter mass distribution is modeled as a _SphericalNFW_.
 - The source `Galaxy`'s light is modeled parametrically as an `EllipticalSersic`.

This uses the pipeline (Check it out full description of the pipeline):

 `autolens_workspace/pipelines/imaging/with_lens_light/light_parametric__mass_mlr_dark__source_parametric.py`.
"""

# %%
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "with_lens_light"
dataset_name = "light_chameleon__mass_mlr_dark__source_sersic"
pixel_scales = 0.1

# %%
"""
Returns the path where the dataset will be loaded from, which in this case is
`/autolens_workspace/dataset/imaging/light_chameleon__mass_mlr_dark__source_sersic`
"""

# %%
dataset_path = f"dataset/{dataset_type}/{dataset_label}/{dataset_name}"

# %%
"""Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""

# %%
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

# %%
"""Next, we create the mask we'll fit this data-set with."""

# %%
mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# %%
"""Make a quick subplot to make sure the data looks as we expect."""

# %%
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
__Settings__

The `SettingsPhaseImaging` describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the `autolens_workspace/examples/model` example scripts, with a 
complete description of all settings given in `autolens_workspace/examples/model/customize/settings.py`.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__Pipeline_Setup__:

Pipelines can contain `Setup` objects.

First, we create a `SetupLightParametric` which customizes:

 - The `LightProfile`'s use to fit different components of the lens light, such as its `bulge` and `disk`.
 - The alignment of these components, for example if the `bulge` and `disk` centres are aligned.
 - If the centre of the lens light profile is manually input and fixed for modeling.
 
In this example we fit the lens light as just one component, a `bulge` represented as an `EllipticalChameleon`. We do 
not fix its centre to an input value. We have included options of `SetupLightParametric` with input values of
`None`, illustrating how it could be edited to fit different models.
"""

# %%
setup_light = al.SetupLightParametric(
    bulge_prior_model=al.lp.EllipticalChameleon,
    disk_prior_model=None,
    envelope_prior_model=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

# %%
"""
This pipeline also uses a `SetupMassLightDark`, which customizes:

 - If there is an `ExternalShear` in the mass model or not.
 - If the centre of the `EllipticalChameleon` `LightMassProfile` and `SphericalNFWMCRLudlow` dark `MassProfile` are 
   aligned.
"""

# %%
setup_mass = al.SetupMassLightDark(align_light_dark_centre=True, with_shear=True)

# %%
"""
Next, we create a `SetupSourceParametric` which does not customize the pipeline behaviour except for tagging (see below).
"""

# %%
setup_source = al.SetupSourceParametric()

# %%
"""
_Pipeline Tagging_

The `Setup` objects are input into a `SetupPipeline` object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if `with_shear` 
is True, the pipeline`s output paths are `tagged` with the string `with_shear`.

This means you can run the same pipeline on the same data twice (e.g. with and without shear) and the results will go
to different output folders and thus not clash with one another!

The `path_prefix` below specifies the path the pipeline results are written to, which is:

 `autolens_workspace/output/transdimensional/dataset_type/dataset_name` 
 `autolens_workspace/output/transdimensional/imaging/light_chameleon__mass_mlr_dark__source_sersic/`

The redshift of the lens and source galaxies are also input (see `examples/model/customize/redshift.py`) for a 
description of what inputting redshifts into **PyAutoLens** does.
"""

# %%
setup = al.SetupPipeline(
    path_prefix=f"transdimensional/{dataset_type}/{dataset_label}/{dataset_name}",
    redshift_lens=0.5,
    redshift_source=1.0,
    setup_light=setup_light,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

# %%
"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
`Setup` and `SettingsPhase` above.
"""

# %%
from autolens_workspace.transdimensional.pipelines.imaging.with_lens_light import (
    light_parametric__mass_light_dark__source_parametric,
)

pipeline = light_parametric__mass_light_dark__source_parametric.make_pipeline(
    setup=setup, settings=settings
)

# %%
"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

# %%
pipeline.run(dataset=imaging, mask=mask)
