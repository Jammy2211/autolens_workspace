"""
__Transdimensional Pipelines__

This transdimensional pipeline runner loads a strong lens dataset and analyses it using a transdimensional lens
modeling pipeline.

This runner and pipeline use **PyAutoLens**`s `hyper-mode`. Hyper-mode passes the best-fit model-image
of previous phases in a pipeline to later phases, and uses these images (called the `hyper images`) to:

- Adapt a pixelization`s grid to the surface-brightness of the source galaxy.
- Adapt the `Regularization` scheme to the surface-brightness of the source galaxy.
- Scale the noise in regions of the image where the model give a poor fit (in both the lens and source galaxies).
- Include uncertanties in the data-reduction into the model, such as the background sky level.

This uses the pipeline (Check it out full description of the pipeline):

 `autolens_workspace/transdimensional/imaging/hyper/pipelines/mass_total__source_inversion.py`.

Check it out now for a detailed description of how it uses the hyper-mode features!
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

"""Next, we create the mask we'll fit this data-set with."""

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

"""Make a quick subplot to make sure the data looks as we expect."""

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

"""
__Settings__

`SettingsPhase` behave as they did in normal pipelines.
"""

settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__PIPELINE SETUP__

Pipelines use `Setup` objects to customize how different aspects of the model are fitted. 

The `SetupHyper` object controls the behaviour of  hyper-mode specifically:

- If hyper-galaxies are used to scale the noise in the lens and source galaxies in image (default False)
- If the level of background noise is modeled throughout the pipeline (default False)
- If the background sky is modeled throughout the pipeline (default False)
"""

setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=False,
    hyper_galaxies_source=False,
    hyper_background_noise=False,
    hyper_image_sky=False,  # <- By default this feature is off, as it rarely changes the lens model.
)

"""
Next, we create a `SetupMassTotal`, which customizes:

 - The `MassProfile` used to fit the lens's total mass distribution.
 - If there is an `ExternalShear` in the mass model or not.
"""

setup_mass = al.SetupMassTotal(
    mass_prior_model=al.mp.EllipticalPowerLaw, with_shear=True
)

"""
In hyper-mode, we can use the `VoronoiBrightnessImage` `Pixelization` and `AdaptiveBrightness` `Regularization` 
scheme, which adapts the `Pixelization` and `Regularization` to the morphology of the lensed source galaxy using the
hyper-image. 

To do this, we create a `SetupSourceInversion` as per usual, passing it these classes. 

We also specify the number of  pixels used by the `Pixelization` to be fixed to 1500 using `inversion_pixel_fixed`. 
This input is optional, a reduced source-resolution can provide faster run-times, but too low a resolution can
lead the source to be poorly reconstructed biasing the mass model. See **HowToLens** chapter 5 for more details.
"""

setup_source = al.SetupSourceInversion(
    pixelization_prior_model=al.pix.VoronoiBrightnessImage,
    regularization_prior_model=al.reg.AdaptiveBrightness,
    inversion_pixels_fixed=1500,
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
 `autolens_workspace/output/transdimensional/imaging/mass_sie__source_inversion/`

The redshift of the lens and source galaxies are also input (see `examples/model/customize/redshift.py`) for a 
description of what inputting redshifts into **PyAutoLens** does.
"""

setup = al.SetupPipeline(
    path_prefix=path.join("hyper", dataset_name),
    setup_hyper=setup_hyper,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

"""
__PIPELINE CREATION__

To create a pipeline we import it from the pipelines folder and run its `make_pipeline` function, inputting the 
`Setup` and `SettingsPhase` above.
"""

from pipelines import mass_total__source_inversion

pipeline = mass_total__source_inversion.make_pipeline(setup=setup, settings=settings)

"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

pipeline.run(dataset=imaging, mask=mask)
