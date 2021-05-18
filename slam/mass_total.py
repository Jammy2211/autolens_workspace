import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Union, Optional, Tuple


def no_lens_light(
    settings_autofit: slam_util.SettingsAutoFit,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_results: af.ResultsCollection,
    mass: af.Model(al.mp.MassProfile) = af.Model(al.mp.EllIsothermal),
    smbh: af.Model(al.mp.MassProfile) = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    end_with_hyper_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SLaM MASS TOTAL PIPELINE for fitting imaging data without a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE or SOURCE INVERSION PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    end_with_hyper_extension
        If `True` a hyper extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images geneted in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the MASS TOTAL PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """
    mass = slam_util.mass__from_result(
        mass=mass, result=source_results.last, unfix_mass_centre=True
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    source = slam_util.source__from_result_model_if_parametric(
        result=source_results.last, setup_hyper=setup_hyper
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_results.last.instance.galaxies.lens.redshift,
                mass=mass,
                smbh=smbh,
                shear=source_results.last.model.galaxies.lens.shear,
            ),
            source=source,
        )
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="mass_total[1]_mass[total]_source",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=100,
    )

    result_1 = search.fit(model=model, analysis=analysis, info=settings_autofit.info)

    """
    __Hyper Extension__

    The above search may be extended with a hyper-search, if the SetupHyper has one or more of the following inputs:

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


def with_lens_light(
    settings_autofit: slam_util.SettingsAutoFit,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_results: af.ResultsCollection,
    light_results: af.ResultsCollection,
    mass: af.Model(al.mp.MassProfile) = af.Model(al.mp.EllIsothermal),
    smbh: af.Model(al.mp.MassProfile) = None,
    mass_centre: Optional[Tuple[float, float]] = None,
    end_with_hyper_extension: bool = False,
) -> af.ResultsCollection:
    """
    The SLaM MASS TOTAL PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE or SOURCE INVERSION PIPELINE which ran before this pipeline.
    light_results
        The results of the SLaM LIGHT PARAMETRIC PIPELINE which ran before this pipeline.
    mass
        The `MassProfile` used to fit the lens galaxy mass in this pipeline.
    smbh
        The `MassProfile` used to fit the a super massive black hole in the lens galaxy.
    mass_centre
       If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
       non-linear search.
    end_with_hyper_extension
        If `True` a hyper extension is performed at the end of the pipeline. If this feature is used, you must be
        certain you have manually passed the new hyper images geneted in this search to the next pipelines.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the MASS TOTAL PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [Priors initialized from SOURCE PIPELINE].
     - The source galaxy's light is parametric or an inversion depending on the previous pipeline [Model and priors 
     initialized from SOURCE PIPELINE].

    This search aims to accurately estimate the lens mass model, using the improved mass model priors and source model 
    of the SOURCE PIPELINE
    """
    mass = slam_util.mass__from_result(
        mass=mass, result=source_results.last, unfix_mass_centre=True
    )

    if mass_centre is not None:
        mass.centre = mass_centre

    if smbh is not None:
        smbh.centre = mass.centre

    source = slam_util.source__from_result_model_if_parametric(
        result=source_results.last, setup_hyper=setup_hyper
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=light_results.last.instance.galaxies.lens.redshift,
                bulge=light_results.last.instance.galaxies.lens.bulge,
                disk=light_results.last.instance.galaxies.lens.disk,
                envelope=light_results.last.instance.galaxies.lens.envelope,
                mass=mass,
                shear=source_results.last.model.galaxies.lens.shear,
                smbh=smbh,
                hyper_galaxy=setup_hyper.hyper_galaxy_lens_from_result(
                    result=light_results.last
                ),
            ),
            source=source,
        )
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="mass_total[1]_light[parametric]_mass[total]_source",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=100,
    )

    result_1 = search.fit(model=model, analysis=analysis, info=settings_autofit.info)

    """
    __Hyper Extension__

    The above search may be extended with a hyper-search, if the SetupHyper has one or more of the following inputs:

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
