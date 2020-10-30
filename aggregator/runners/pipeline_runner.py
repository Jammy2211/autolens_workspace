"""
__Aggregator: Pipeline Runner__

This script fits the sample of three strong lenses simulated by the script `autolens_workspace/aggregator/sample.py`
using a pipeline to illustrate aggregator functionality in the tutorial:

 - a5_pipelines
 - a6_advanced
 
If you are not yet familiar with PyAutoLens`s pipeline functionality, you should checkout
`autolens_workspace/pipelines` and `howtolens/chapter_3_pipelines` before doing these tutorials.

Using a pipeline composed of three phases this runner fits `Imaging` of a strong lens system, where:
 
 - An `EllipticalIsothermal` `MassProfile`.for the lens `Galaxy`'s mass.
 - An `Inversion` for the source `Galaxy`'s light.
"""

import autolens as al

"""Specify the dataset type, label and name, which we use to determine the path we load the data from."""

pixel_scales = 0.1

for dataset_name in [
    "mass_sie__source_bulge__0",
    "mass_sie__source_bulge__1",
    "mass_sie__source_bulge__2",
]:

    """Set up the config and output paths."""
    dataset_path = f"dataset/aggregator/{dataset_name}"

    """Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""
    imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
    )

    """Next, we create the mask we'll fit this data-set with."""
    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    # %%
    """
    The `SettingsPhase` (which customize the fit of the phase`s fit), will also be available to the aggregator!
    """

    # %%
    settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

    settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

    # %%
    """
    __Pipeline_Setup_And_Tagging__:

    For this pipeline the pipeline setup customizes:

  - The `Pixelization` used by the `Inversion` of this pipeline.
  - The `Regularization` scheme used by the `Inversion` of this pipeline.
  - If there is an `ExternalShear` in the mass model or not.
    """

    setup_mass = al.SetupMassTotal(with_shear=False)

    setup_source = al.SetupSourceInversion(
        pixelization_prior_model=al.pix.VoronoiMagnification,
        regularization_prior_model=al.reg.Constant,
    )

    setup = al.SetupPipeline(
        path_prefix=f"aggregator/{dataset_name}",
        setup_mass=setup_mass,
        setup_source=setup_source,
    )

    from pipelines import mass_total__source_inversion

    pipeline = mass_total__source_inversion.make_pipeline(
        setup=setup, settings=settings
    )

    pipeline.run(dataset=imaging, mask=mask)
