# This script fits the sample of three strong lenses simulated by the script `autolens_workspace/aggregator/sample.py`
# using a beginner pipeline to illustrate aggregator functionality.

# we fit each lens with an  `EllipticalIsothermal` `MassProfile`.and each source using a pixelized `Inversion`. The fit will use a
# beginner pipelines which performs a 3 phase analysis, which will allow us to illustrate how the results of different
# phases can be loaded using the aggregator.

# This script follows the scripts described in `autolens_workspace/runners/beginner/` and the pipelines:

# `autolens_workspace/pipelines/beginner/no_lens_light/mass_total__source_inversion.py`

# If anything doesn`t make sense check those scripts out for details!
import autolens as al

"""Specify the dataset type, label and name, which we use to determine the path we load the data from."""

pixel_scales = 0.1

for dataset_name in [
    "mass_sie__source_bulge__0",
    "mass_sie__source_bulge__1",
    "mass_sie__source_bulge__2",
]:

    # Create the path where the dataset will be loaded from, which in this case is
    # `/autolens_workspace/aggregator/dataset/imaging/mass_sie__source_sersic`
    dataset_path = f"aggregator/dataset/{dataset_name}"

    ### Info ###

    # The dataset name and info are accessible to the aggregator, to aid interpretation of results. The name is passed
    # as a string and info as a dictionary.

    name = dataset_name

    info = {
        "setup.redshift_lens": 0.5,
        "setup.redshift_source": 1.0,
        "velocity_dispersion": 250000,
        "stellar mass": 1e11,
    }

    ### DATASET ###

    """Using the dataset path, load the data (image, noise-map, PSF) as an `Imaging` object from .fits files."""
    imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
        positions_path=f"{dataset_path}/positions.dat",
        name=name,
    )

    """Next, we create the mask we'll fit this data-set with."""
    mask = al.Mask2D.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    ### PIPELINE SETUP ###

    # Advanced pipelines use the same `Source`, `Light` and `Mass` setup objects we used in beginner and intermediate
    # pipelines. However, there are many additional options now available with these setup objects, that did not work
    # for beginner and intermediate pipelines. For an explanation, checkout:

    # - `autolens_workspace/runners/advanced/doc_setup`

    # The setup of earlier pipelines inform the model fitted in later pipelines. For example:

    # - The `Pixelization` and `Regularization` scheme used in the source (inversion) pipeline will be used in the light and
    #   mass pipelines.

    hyper = al.SetupHyper(
        hyper_galaxies_lens=False,
        hyper_galaxies_source=False,
        hyper_image_sky=False,
        hyper_background_noise=False,
    )

    source = al.SLaMPipelineSource(with_shear=False)

    mass = al.SLaMPipelineMass(with_shear=False)

    setup = al.SLaM(setup_hyper=hyper, source=source, pipeline_mass=mass)

    # We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!

    ### SOURCE ###

    from pipelines.advanced.no_lens_light.source.parametric import (
        mass_sie__source_sersic,
    )

    pipeline_source__parametric = mass_sie__source_sersic.make_pipeline(
        setup=setup,
        path_prefix=f"aggregator/grid_search/{dataset_name}",
        positions_threshold=1.0,
    )

    ### MASS ###

    from pipelines.advanced.no_lens_light.mass.sie import mass_sie__source

    pipeline_mass__sie = mass_sie__source.make_pipeline(
        setup=setup,
        path_prefix=f"aggregator/grid_search/{dataset_name}",
        positions_threshold=1.0,
    )

    ### SUBHALO ###

    from pipelines.advanced.no_lens_light.subhalo import lens_mass__subhalo_nfw__source

    pipeline_subhalo__nfw = lens_mass__subhalo_nfw__source.make_pipeline(
        setup=setup,
        path_prefix=f"aggregator/grid_search/{dataset_name}",
        positions_threshold=1.0,
        grid_size=2,
        parallel=False,
    )

    ### PIPELINE COMPOSITION AND RUN ###

    # We finally add the pipelines above together, which means that they will run back-to-back, as usual passing
    # information throughout the analysis to later phases.

    pipeline = pipeline_source__parametric + pipeline_mass__sie + pipeline_subhalo__nfw

    pipeline.run(dataset=imaging, mask=mask)
