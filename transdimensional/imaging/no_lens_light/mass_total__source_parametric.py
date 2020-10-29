"""
__TRANSDIMENSIONAL PIPELINES__

This transdimensional pipeline runner loads a strong lens dataset and analyses it using a transdimensional lens
modeling pipeline.

Using a pipeline composed of two phases this runner fits `Imaging` of a strong lens system, where in the final phase
of the pipeline:

 - The lens `Galaxy`'s light is omitted from the data and model.
 - The lens `Galaxy`'s total mass distribution is modeled as an `EllipticalIsothermal`.
 - The source `Galaxy`'s two `LightProfile`'s are modeled as `EllipticalSersic``..

This uses the pipeline (Check it out full description of the pipeline):
"""

import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__source_sersic"
pixel_scales = 0.1

dataset_path = f"dataset/imaging/no_lens_light/{dataset_name}"

"""Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

"""Next, we create the mask we'll fit this data-set with."""

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.4
)

"""Make a quick subplot to make sure the data looks as we expect."""

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
__Pipeline_Setup__:

Pipelines use `Setup` objects to customize how different aspects of the model are fitted. 

First, we create a `SetupMassTotal`, which customizes:

 - The `MassProfile` used to fit the lens's total mass distribution.
 - If there is an `ExternalShear` in the mass model or not (this lens was not simulated with shear and 
   we do not include it in the mass model).
"""

setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalPowerLaw, with_shear=False
)

"""
Next, we create a `SetupSourceParametric` which customizes:

 - The `LightProfile`'s which fit different components of the source light, such as its `bulge` and `disk`.
 - The alignment of these components, for example if the `bulge` and `disk` centres are aligned.
 
In this example we fit the source light as one component, a `bulge` represented as an `EllipticalSersic`. We have 
included options of `SetupSourceParametric` with input values of `None`, illustrating how it could be edited to fit different models.
"""

setup_source = al.SetupSourceParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=None,
    envelope_prior_model=None,
    align_bulge_disk_centre=False,
    align_bulge_disk_elliptical_comps=False,
)

"""
_Pipeline Tagging_

The `Setup` objects are input into a `SetupPipeline` object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if `with_shear` 
is True, the pipeline`s output paths are `tagged` with the string `with_shear`.

This means you can run the same pipeline on the same data twice (e.g. with and without shear) and the results will go
to different output folders and thus not clash with one another!

The `path_prefix` below specifies the path the pipeline results are written to, which is:

 `autolens_workspace/output/transdimensional/dataset_type/dataset_name` 
 `autolens_workspace/output/transdimensional/imaging/mass_sie__source_sersic/`
 
 The redshift of the lens and source galaxies are also input (see `examples/model/customize/redshift.py`) for a 
description of what inputting redshifts into **PyAutoLens** does.
"""

setup = al.SetupPipeline(
    path_prefix=f"transdimensional/{dataset_name}",
    redshift_lens=0.5,
    redshift_source=1.0,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
`Setup` and `SettingsPhase` above.
"""

from pipelines import (
    mass_total__source_parametric,
)

pipeline = mass_total__source_parametric.make_pipeline(setup=setup, settings=settings)

"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

pipeline.run(dataset=imaging, mask=mask)
