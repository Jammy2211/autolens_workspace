import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Union, Optional


def with_lens_light(
    settings_autofit: slam_util.SettingsAutoFit,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_results: af.ResultsCollection,
    lens_bulge: af.Model = af.Model(al.lp.EllSersic),
    lens_disk: Optional[af.Model] = None,
    lens_envelope: Optional[af.Model] = None,
    end_with_hyper_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SlaM LIGHT PARAMETRIC PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    settings_autofit
        A collection of settings that control the behaviour of PyAutoFit thoughout the pipeline (e.g. paths, database,
        parallelization, etc.).
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE or SOURCE INVERSION PIPELINE which ran before this pipeline.
    lens_bulge
        The `LightProfile` `Model` used to represent the light distribution of the lens galaxy's bulge (set to
        None to omit a bulge).
    lens_disk
        The `LightProfile` `Model` used to represent the light distribution of the lens galaxy's disk (set to
        None to omit a disk).
    lens_envelope
        The `LightProfile` `Model` used to represent the light distribution of the lens galaxy's envelope (set to
        None to omit an envelope).
    end_with_hyper_extension
        If `True` a hyper extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images geneted in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the LIGHT PARAMETRIC PIPELINE we fit a lens model where:

     - The lens galaxy light is modeled using parametric bulge + disk + envelope [no prior initialization].
     - The lens galaxy mass is modeled using SOURCE PIPELINE's mass distribution [Parameters fixed from MASS PIPELINE].
     - The source galaxy's light is modeled using SOURCE PIPELINE's model [Parameters fixed from SOURCE PIPELINE].

    This search aims to produce an accurate model of the lens galaxy's light, which may not have been possible in the
    SOURCE PIPELINE as the mass and source models were not properly initialized.
    """

    """
    If hyper-galaxy noise scaling for the lens is on, it may have scaled the noise to high values in the SOURCE
    PIPELINE (which fitted a simpler lens light model than this pipeline). The new lens light model fitted in this
    pipeline may fit the data better, requiring a reducing level of noise scaling. For this reason, the noise scaling
    normalization is included as a free parameter.
    """
    hyper_galaxy = setup_hyper.hyper_galaxy_lens_from_result(
        result=source_results.last, noise_factor_is_model=True
    )

    source = slam_util.source__from_result(
        result=source_results.last, setup_hyper=setup_hyper, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_results.last.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=lens_disk,
                envelope=lens_envelope,
                mass=source_results.last.instance.galaxies.lens.mass,
                shear=source_results.last.instance.galaxies.lens.shear,
                hyper_galaxy=hyper_galaxy,
            ),
            source=source,
        ),
        hyper_image_sky=setup_hyper.hyper_image_sky_from_result(
            result=source_results.last, as_model=True
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from_result(
            result=source_results.last
        ),
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="light[1]_light[parametric]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=150,
    )

    result_1 = search.fit(
        model=model, analysis=analysis.no_positions, info=settings_autofit.info
    )

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

     - The source is using an `Inversion`.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """

    if end_with_hyper_extension:

        result_1 = extensions.hyper_fit(
            setup_hyper=setup_hyper,
            result=result_1,
            analysis=analysis,
            include_hyper_image_sky=True,
        )

    return af.ResultsCollection([result_1])
