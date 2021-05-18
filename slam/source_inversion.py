import autofit as af
import autolens as al
from . import slam_util
from . import extensions

from typing import Union, Optional


def no_lens_light(
    settings_autofit: slam_util.SettingsAutoFit,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_parametric_results: af.ResultsCollection,
    pixelization: af.Model(al.pix.Pixelization) = af.Model(
        al.pix.VoronoiBrightnessImage
    ),
    regularization: af.Model(al.reg.Regularization) = af.Model(al.reg.Constant),
) -> af.ResultsCollection:
    """
    The S:aM SOURCE INVERSION PIPELINE for fitting imaging data without a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_parametric_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE which ran before this pipeline.
    pixelization
        The pixelization used by the `Inversion` which fits the source light.
    regularization
        The regularization used by the `Inversion` which fits the source light.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of SOURCE PARAMETRIC 
     PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme.

    This search aims to quickly estimate values for the pixelization resolution and regularization coefficient.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.lens.redshift,
                mass=source_parametric_results.last.instance.galaxies.lens.mass,
                shear=source_parametric_results.last.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.source.redshift,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=setup_hyper.hyper_galaxy_source_from_result(
                    result=source_parametric_results.last
                ),
            ),
        ),
        hyper_image_sky=setup_hyper.hyper_image_sky_from_result(
            result=source_parametric_results.last, as_model=False
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from_result(
            result=source_parametric_results.last
        ),
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[1]_mass[fixed]_source[inversion_magnification_initialization]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=30,
    )

    result_1 = search.fit(
        model=model, analysis=analysis.no_positions, info=settings_autofit.info
    )

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    In search 2 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE PARAMETRIC PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme 
     [parameters are fixed to the result of search 1].

    This search aims to improve the lens mass model using the search 1 `Inversion`.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.lens.redshift,
                mass=source_parametric_results.last.model.galaxies.lens.mass,
                shear=source_parametric_results.last.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.source.redshift,
                pixelization=result_1.instance.galaxies.source.pixelization,
                regularization=result_1.instance.galaxies.source.regularization,
                hyper_galaxy=result_1.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_1.instance.hyper_image_sky,
        hyper_background_noise=result_1.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[2]_mass[total]_source[fixed]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=50,
    )

    result_2 = search.fit(model=model, analysis=analysis, info=settings_autofit.info)

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    In search 3 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
     - The source galaxy's light is the input pixelization and regularization.

    This search aims to estimate values for the pixelization and regularization scheme.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.lens.redshift,
                mass=result_2.instance.galaxies.lens.mass,
                shear=result_2.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.source.redshift,
                pixelization=pixelization,
                regularization=regularization,
                hyper_galaxy=result_2.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_2.instance.hyper_image_sky,
        hyper_background_noise=result_2.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[3]_mass[fixed]_source[inversion_initialization]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=30,
        dlogz=10.0,
        sample="rstagger",
    )

    analysis.set_hyper_dataset(result=result_2)

    result_3 = search.fit(
        model=model, analysis=analysis.no_positions, info=settings_autofit.info
    )
    result_3.use_as_hyper_dataset = True

    """
    __Model + Search + Analysis + Model-Fit (Search 4)__

    In search 4 of the SOURCE INVERSION PIPELINE we fit a lens model where:

     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     search 2].
     - The source galaxy's light is the input pixelization and regularization scheme [parameters fixed to the result 
     of search 3].

    This search aims to improve the lens mass model using the input `Inversion`.
    """
    mass = slam_util.mass__from_result(
        mass=result_2.model.galaxies.lens.mass,
        result=source_parametric_results.last,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.lens.redshift,
                mass=mass,
                shear=result_2.model.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.source.redshift,
                pixelization=result_3.instance.galaxies.source.pixelization,
                regularization=result_3.instance.galaxies.source.regularization,
                hyper_galaxy=result_3.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_3.instance.hyper_image_sky,
        hyper_background_noise=result_3.instance.hyper_background_noise,
    )

    analysis.preloads = al.Preloads.setup(
        result=result_3, model=model, pixelization=True
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[4]_mass[total]_source[fixed]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=50,
    )

    result_4 = search.fit(model=model, analysis=analysis, info=settings_autofit.info)

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

     - The source is using an `Inversion`.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """
    result_4 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_4,
        analysis=analysis,
        include_hyper_image_sky=True,
    )

    return af.ResultsCollection([result_1, result_2, result_3, result_4])


def with_lens_light(
    settings_autofit: slam_util.SettingsAutoFit,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    setup_hyper: al.SetupHyper,
    source_parametric_results: af.ResultsCollection,
    pixelization: af.Model(al.pix.Pixelization) = af.Model(
        al.pix.VoronoiBrightnessImage
    ),
    regularization: af.Model(al.reg.Regularization) = af.Model(al.reg.Constant),
) -> af.ResultsCollection:
    """
    The SLaM SOURCE INVERSION PIPELINE for fitting imaging data with a lens light component.

    Parameters
    ----------
    analysis
        The analysis class which includes the `log_likelihood_function` and can be customized for the SLaM model-fit.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_parametric_results
        The results of the SLaM SOURCE PARAMETRIC PIPELINE which ran before this pipeline.
    pixelization
        The pixelization used by the `Inversion` which fits the source light.
    regularization
        The regularization used by the `Inversion` which fits the source light.
    """

    """
    __Model + Search + Analysis + Model-Fit (Search 1)__

    In search 1 of the SOURCE INVERSION PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of SOURCE PARAMETRIC 
     PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme.

    This search aims to quickly estimate values for the pixelization resolution and regularization coefficient.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.lens.redshift,
                bulge=source_parametric_results.last.instance.galaxies.lens.bulge,
                disk=source_parametric_results.last.instance.galaxies.lens.disk,
                envelope=source_parametric_results.last.instance.galaxies.lens.envelope,
                mass=source_parametric_results.last.instance.galaxies.lens.mass,
                shear=source_parametric_results.last.instance.galaxies.lens.shear,
                hyper_galaxy=setup_hyper.hyper_galaxy_lens_from_result(
                    result=source_parametric_results.last
                ),
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_parametric_results.last.instance.galaxies.source.redshift,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=setup_hyper.hyper_galaxy_source_from_result(
                    result=source_parametric_results.last
                ),
            ),
        ),
        hyper_image_sky=setup_hyper.hyper_image_sky_from_result(
            result=source_parametric_results.last, as_model=False
        ),
        hyper_background_noise=setup_hyper.hyper_background_noise_from_result(
            result=source_parametric_results.last
        ),
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[1]_light[fixed]_mass[fixed]_source[inversion_magnification_initialization]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=30,
    )

    result_1 = search.fit(
        model=model, analysis=analysis.no_positions, info=settings_autofit.info
    )

    """
    __Model + Search + Analysis + Model-Fit (Search 2)__

    In search 2 of the SOURCE INVERSION PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     SOURCE PARAMETRIC PIPELINE].
     - The source galaxy's light is a `VoronoiMagnification` pixelization and `Constant` regularization scheme 
     [parameters are fixed to the result of search 1].

    This search aims to improve the lens mass model using the search 1 `Inversion`.
    """
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.lens.redshift,
                bulge=result_1.instance.galaxies.lens.bulge,
                disk=result_1.instance.galaxies.lens.disk,
                envelope=result_1.instance.galaxies.lens.envelope,
                mass=source_parametric_results.last.model.galaxies.lens.mass,
                shear=source_parametric_results.last.model.galaxies.lens.shear,
                hyper_galaxy=result_1.instance.galaxies.lens.hyper_galaxy,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_1.instance.galaxies.source.redshift,
                pixelization=result_1.instance.galaxies.source.pixelization,
                regularization=result_1.instance.galaxies.source.regularization,
                hyper_galaxy=result_1.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_1.instance.hyper_image_sky,
        hyper_background_noise=result_1.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[2]_light[fixed]_mass[total]_source[inversion_magnification]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=50,
    )

    result_2 = search.fit(model=model, analysis=analysis, info=settings_autofit.info)

    """
    __Model + Search + Analysis + Model-Fit (Search 3)__

    In search 3 of the SOURCE INVERSION PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters fixed to result of search 2].
     - The source galaxy's light is the input pixelization and regularization.

    This search aims to estimate values for the pixelization and regularization scheme.
    """

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.lens.redshift,
                bulge=result_2.instance.galaxies.lens.bulge,
                disk=result_2.instance.galaxies.lens.disk,
                envelope=result_2.instance.galaxies.lens.envelope,
                mass=result_2.instance.galaxies.lens.mass,
                shear=result_2.instance.galaxies.lens.shear,
                hyper_galaxy=result_2.instance.galaxies.lens.hyper_galaxy,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_2.instance.galaxies.source.redshift,
                pixelization=pixelization,
                regularization=regularization,
                hyper_galaxy=result_2.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_2.instance.hyper_image_sky,
        hyper_background_noise=result_2.instance.hyper_background_noise,
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[3]_light[fixed]_mass[fixed]_source[inversion_initialization]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=30,
        dlogz=10.0,
        sample="rstagger",
    )

    analysis.set_hyper_dataset(result=result_2)

    result_3 = search.fit(
        model=model, analysis=analysis.no_positions, info=settings_autofit.info
    )
    result_3.use_as_hyper_dataset = True

    """
    __Model + Search + Analysis + Model-Fit (Search 4)__

    In search 4 of the SOURCE INVERSION PIPELINE we fit a lens model where:

    - The lens galaxy light is modeled using a parametric bulge + disk + envelope [parameters fixed to result of SOURCE
    PARAMETER PIPELINE].
     - The lens galaxy mass is modeled using a total mass distribution [parameters initialized from the results of the 
     search 2].
     - The source galaxy's light is the input pixelization and regularization scheme [parameters fixed to the result 
     of search 3].

    This search aims to improve the lens mass model using the input `Inversion`.
    """
    mass = slam_util.mass__from_result(
        mass=result_2.model.galaxies.lens.mass,
        result=source_parametric_results.last,
        unfix_mass_centre=True,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.lens.redshift,
                bulge=result_3.instance.galaxies.lens.bulge,
                disk=result_3.instance.galaxies.lens.disk,
                envelope=result_3.instance.galaxies.lens.envelope,
                mass=mass,
                shear=result_2.model.galaxies.lens.shear,
                hyper_galaxy=result_3.instance.galaxies.lens.hyper_galaxy,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=result_3.instance.galaxies.source.redshift,
                pixelization=result_3.instance.galaxies.source.pixelization,
                regularization=result_3.instance.galaxies.source.regularization,
                hyper_galaxy=result_3.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=result_3.instance.hyper_image_sky,
        hyper_background_noise=result_3.instance.hyper_background_noise,
    )

    analysis.preloads = al.Preloads.setup(
        result=result_3, model=model, pixelization=True
    )

    search = af.DynestyStatic(
        path_prefix=settings_autofit.path_prefix,
        name="source_inversion[4]_light[fixed]_mass[total]_source[inversion]",
        unique_tag=settings_autofit.unique_tag,
        number_of_cores=settings_autofit.number_of_cores,
        session=settings_autofit.session,
        nlive=50,
    )

    result_4 = search.fit(model=model, analysis=analysis, info=settings_autofit.info)

    """
    __Hyper Extension__

    The above search is extended with a hyper-search if the SetupHyper has one or more of the following inputs:

     - The source is using an `Inversion`.
     - One or more `HyperGalaxy`'s are included.
     - The background sky is included via `hyper_image_sky` input.
     - The background noise is included via the `hyper_background_noise`.
    """
    result_4 = extensions.hyper_fit(
        setup_hyper=setup_hyper,
        result=result_4,
        analysis=analysis,
        include_hyper_image_sky=True,
    )

    return af.ResultsCollection([result_1, result_2, result_3, result_4])
