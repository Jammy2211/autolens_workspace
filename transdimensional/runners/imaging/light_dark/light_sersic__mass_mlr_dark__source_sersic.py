# %%
"""
__WELCOME__ 

This transdimensional pipeline runner loads a strong lens dataset and analyses it using a transdimensional lens
modeling pipeline.

Using a pipeline composed of five phases this runner fits `Imaging` of a strong lens system, where in the final phase
of the pipeline:

 - The lens galaxy`s `LightProfile` is modeled as an _EllipticalSersic_.
 - The lens galaxy`s stellar `MassProfile` is fitted with the `EllipticalSersic` of the
      `LightProfile`, where it is converted to a stellar mass distribution via a constant mass-to-light ratio.
 - The lens galaxy`s dark matter `MassProfile` is modeled as a _SphericalNFW_.
 - The source galaxy`s `LightProfile` is modeled as an _EllipticalSersic_.

This uses the pipeline (Check it out full description of the pipeline):

 `autolens_workspace/pipelines/imaging/with_lens_light/light_sersic__mass_mlr_dark__source_sersic.py`.
"""

# %%
"""Use the WORKSPACE environment variable to determine the path to the `autolens_workspace`."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "with_lens_light"
dataset_name = "light_sersic__mass_mlr_dark__source_sersic"
pixel_scales = 0.1

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
`/autolens_workspace/dataset/imaging/light_sersic__mass_mlr_dark__source_sersic`
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
Due to the slow deflection angle calculation of the `EllipticalSersic` `MassProfile` we use `GridInterpolate` 
objects to speed up the analysis.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(
    grid_class=al.GridInterpolate, pixel_scales_interp=0.1
)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__Pipeline_Setup__:

Pipelines can contain `Setup` objects.

First, we create a `SetupLightSersic` which does not customize the pipeline behaviour except for tagging (see below)
"""

# %%
setup_light = al.SetupLightSersic()

# %%
"""
This pipeline also uses a `SetupMassLightDark`, which customizes:

 - If there is an `ExternalShear` in the mass model or not.
 - If the centre of the `EllipticalSersic` `LightMassProfile` and `SphericalNFWMCRLudlow` dark `MassProfile` are 
   aligned.
"""

# %%
setup_mass = al.SetupMassLightDark(align_light_dark_centre=True, no_shear=False)

# %%
"""
Next, we create a `SetupSourceSersic` which does not customize the pipeline behaviour except for tagging (see below).
"""

# %%
setup_source = al.SetupSourceSersic()

# %%
"""
_Pipeline Tagging_

The `Setup` objects are input into a `SetupPipeline` object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if `no_shear` 
is True, the pipeline`s output paths are `tagged` with the string `no_shear`.

This means you can run the same pipeline on the same data twice (with and without shear) and the results will go
to different output folders and thus not clash with one another!

The `folders` below specify the path the pipeline results are written to, which is:

 `autolens_workspace/output/pipelines/dataset_type/dataset_name` 
 `autolens_workspace/output/pipelines/imaging/light_sersic__mass_mlr_dark__source_inversion/`

The redshift of the lens and source galaxies are also input (see `examples/model/customimze/redshift.py`) for a 
description of what inputting redshifts into **PyAutoLens** does.
"""

# %%
setup = al.SetupPipeline(
    folders=["transdimensional", f"{dataset_type}_{dataset_label}", dataset_name],
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
*Setup* and *SettingsPhase* above.
"""

# %%
from autolens_workspace.transdimensional.pipelines.imaging.light_dark import (
    light_sersic__mass_mlr_dark__source_sersic,
)

pipeline = light_sersic__mass_mlr_dark__source_sersic.make_pipeline(
    setup=setup, settings=settings
)

# %%
"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

# %%
pipeline.run(dataset=imaging, mask=mask)
