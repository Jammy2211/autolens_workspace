"""
__Aggregator: Phase Runner__

This script fits a sample of three strong lenses simulated by the script `autolens_workspace/aggregator/sample.py`
using a single `PhaseImaging` object, to illustrate aggregator functionality in the tutorials:

 - a1_samples
 - a2_lens_models
 - a3_data_fitting
 - a4_derived

The phase fits each lens with:
 
 - An `EllipticalIsothermal` `MassProfile`.for the lens `Galaxy`'s mass.
 - An `EllipticalSersic` `LightProfile`.for the source `Galaxy`'s light.
"""

import autofit as af
import autolens as al

"""Specify the dataset type, label and name, which we use to determine the path we load the data from."""

pixel_scales = 0.1

for dataset_name in [
    "mass_sie__source_sersic__0",
    "mass_sie__source_sersic__1",
    "mass_sie__source_sersic__2",
]:

    """Set up the config and output paths."""
    dataset_path = f"dataset/aggregator/{dataset_name}"

    """
    Info:

    We can pass information on our dataset to the `phase.run()` method, which will be accessible to the aggregator 
    to aid interpretation of results. This information is passed as a dictionary, with th redshifts of the lens
    and source good examples of information you may wish to pass.
    """
    info = {
        "redshift_lens": 0.5,
        "redshift_source": 1.0,
        "velocity_dispersion": 250000,
        "stellar mass": 1e11,
    }

    """
    Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files.
    
    This `Imaging` object will be available via the aggregator. Note also that we give the dataset a `name` via the
    command `name=dataset_name`. we'll use this name in the aggregator tutorials.
    """
    imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
        name=dataset_name,
    )

    """The `Mask2D` we fit this data-set with, which will be available via the aggregator."""
    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    """
    Pickle Files:

    We can pass strings specifying the path and filename of .pickle files stored on our hard-drive to the `phase.run()`
    method, which will make them accessible to the aggregator  to aid interpretation of results. Our simulated strong
    lens datasets have a `true_tracer.pickle` file which we pass in below, which we use in the `Aggregator` tutorials to
    easily illustrate how we can check if a model-fit recovers its true input parameters.
    """
    pickle_files = [f"{dataset_path}/true_tracer.pickle"]

    # %%
    """
    The `SettingsPhase` (which customize the fit of the phase`s fit), will also be available to the aggregator!
    """

    # %%
    settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

    settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

    phase = al.PhaseImaging(
        search=af.DynestyStatic(
            path_prefix=f"aggregator/phase_runner/{dataset_name}",
            name="phase_aggregator",
            n_live_points=50,
        ),
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic),
        ),
        settings=settings,
    )

    phase.run(dataset=imaging, mask=mask, info=info, pickle_files=pickle_files)
