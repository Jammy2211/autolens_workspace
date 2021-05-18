import autofit as af
import autolens as al

from typing import Tuple, Optional
from sqlalchemy.orm import Session


class SettingsAutoFit:
    def __init__(
        self,
        path_prefix: str,
        unique_tag: Optional[str] = None,
        info: Optional[dict] = None,
        number_of_cores: Optional[int] = 1,
        session: Optional[Session] = None,
    ):
        """
        The settings of PyAutoFit that are used throughout a SLaM pipeline.

        Parameters
        ----------
        path_prefix
            The prefix of folders between the output path and the search folders.
        unique_tag
            The unique tag for this model-fit, which will be given a unique entry in the sqlite database and also acts as
            the folder after the path prefix and before the search name. This is typically the name of the dataset.
        info : dict
            Optional dictionary containing information about the model-fit that is stored in the database and can be
            loaded by the aggregator after the model-fit is complete.
        number_of_cores
            The number of CPU cores used to parallelize the model-fit. This is used internally in a non-linear search
            for most model fits, but is done on a per-fit basis for grid based searches (e.g. sensitivity mapping).
        session
            The SQLite database session which is active means results are directly wrtten to the SQLite database
            at the end of a fit and loaded from the database at the start.
        """

        self.path_prefix = path_prefix
        self.unique_tag = unique_tag
        self.info = info
        self.number_of_cores = number_of_cores
        self.session = session


def set_lens_light_centres(lens, light_centre: Tuple[float, float]):
    """
    Set the (y,x) centre of every light profile in the lens light model to the same input value `light_centre`
    Parameters
    ----------
    lens : af.Model(al.Galaxy)
        The `Galaxy` containing the light models of the distribution of the lens galaxy's bulge, disk and envelope.
    light_centre : (float, float) or None
       If input, the centre of every light model centre is set using this (y,x) value.
    """

    if lens.bulge is not None:
        lens.bulge.centre = light_centre

    if lens.disk is not None:
        lens.disk.centre = light_centre

    if lens.envelope is not None:
        lens.envelope.centre = light_centre


def set_lens_light_model_centre_priors(
    lens: af.Model, light_centre_gaussian_prior_values: Tuple[float, float]
):
    """
    Set the mean and sigma of every `GaussianPrior` of every light profile in the lens light model to the same value,
    for the y and x coordinates.

    This can be used to specifically customize only the prior on the lens light model centre, given that in many
    datasets this is clearly visible by simply looking at the image itself.

    Parameters
    ----------
    lens : af.Model(al.Galaxy)
        The `Galaxy` containing the light models of the distribution of the lens galaxy's bulge, disk and envelope.
    light_centre_gaussian_prior_values : (float, float) or None
       If input, the mean and sigma of every light model centre is set using these values as (mean, sigma).
    """

    mean = light_centre_gaussian_prior_values[0]
    sigma = light_centre_gaussian_prior_values[1]

    if lens.bulge is not None:
        lens.bulge.centre_0 = af.GaussianPrior(mean=mean, sigma=sigma)
        lens.bulge.centre_1 = af.GaussianPrior(mean=mean, sigma=sigma)

    if lens.disk is not None:
        lens.disk.centre_0 = af.GaussianPrior(mean=mean, sigma=sigma)
        lens.disk.centre_1 = af.GaussianPrior(mean=mean, sigma=sigma)

    if lens.envelope is not None:
        lens.envelope.centre_0 = af.GaussianPrior(mean=mean, sigma=sigma)
        lens.envelope.centre_1 = af.GaussianPrior(mean=mean, sigma=sigma)


def pass_light_and_mass_profile_priors(
    model: af.Model(al.lmp.LightMassProfile),
    result_light_component: af.Model,
    result: af.Result,
    einstein_mass_range: Optional[Tuple[float, float]] = None,
    as_instance: bool = False,
) -> Optional[af.Model]:
    """
    Returns an updated version of a `LightMassProfile` model (e.g. a bulge or disk) whose priors are initialized from
    previous results of a `Light` pipeline.

    This function generically links any `LightProfile` to any `LightMassProfile`, pairing parameters which share the
    same path.

    It also allows for an Einstein mass range to be input, such that the `LogUniformPrior` on the mass-to-light
    ratio of the model-component is set with lower and upper limits that are a multiple of the Einstein mass
    computed in the previous SOURCE PIPELINE. For example, if `einstein_mass_range=[0.01, 5.0]` the mass to light
    ratio will use priors corresponding to values which give Einstein masses 1% and 500% of the estimated Einstein mass.

    Parameters
    ----------
    model : af.Model(al.lmp.LightMassProfile)
        The light and mass profile whoses priors are passed from the LIGHT PIPELINE.
    result_light_component : af.Result
        The `LightProfile` result of the LIGHT PIPELINE used to pass the priors.
    result : af.Result
        The result of the LIGHT PIPELINE used to pass the priors.
    einstein_mass_range : (float, float)
        The values a the estimate of the Einstein Mass in the LIGHT PIPELINE is multiplied by to set the lower and
        upper limits of the profile's mass-to-light ratio.
    as_instance : bool
        If `True` the prior is set up as an instance, else it is set up as a model component.

    Returns
    -------
    af.Model(mp.LightMassProfile)
        The light and mass profile whose priors are initialized from a previous result.
    """

    if model is None:
        return model

    model.take_attributes(source=result_light_component)

    if einstein_mass_range is not None:

        model = update_mass_to_light_ratio_prior(
            model=model, result=result, einstein_mass_range=einstein_mass_range
        )

    return model


def update_mass_to_light_ratio_prior(
    model: af.Model(al.lmp.LightMassProfile),
    result: af.Result,
    einstein_mass_range: Tuple[float, float],
    bins: int = 100,
) -> Optional[af.Model]:
    """
    Updates the mass to light ratio parameter of a `LightMassProfile` model (e.g. a bulge or disk) such that the
    the `LogUniformPrior` on the mass-to-light ratio of the model-component is set with lower and upper limits that
    are a multiple of the Einstein mass computed in the previous SOURCE PIPELINE.

    For example, if `einstein_mass_range=[0.01, 5.0]` the mass to light ratio will use priors corresponding to
    values which give Einstein masses 1% and 500% of the estimated Einstein mass.

    Parameters
    ----------
    model
        The light and mass profile whoses priors are passed from the LIGHT PIPELINE.
    result
        The result of the LIGHT PIPELINE used to pass the priors.
    einstein_mass_range
        The values a the estimate of the Einstein Mass in the LIGHT PIPELINE is multiplied by to set the lower and upper
        limits of the profile's mass-to-light ratio.
    bins
        The number of bins used to map a calculated Einstein Mass to that of the `LightMassProfile`.

    Returns
    -------
    af.Model(mp.LightMassProfile)
        The light and mass profile whose mass-to-light ratio prior is set using the input Einstein mass and range.
    """

    if model is None:
        return None

    grid = result.max_log_likelihood_fit.grid

    einstein_radius = result.max_log_likelihood_tracer.einstein_radius_from_grid(
        grid=grid
    )

    einstein_mass = result.max_log_likelihood_tracer.einstein_mass_angular_from_grid(
        grid=grid
    )

    einstein_mass_lower = einstein_mass_range[0] * einstein_mass
    einstein_mass_upper = einstein_mass_range[1] * einstein_mass

    instance = model.instance_from_prior_medians()

    mass_to_light_ratio_lower = instance.normalization_from_mass_angular_and_radius(
        mass_angular=einstein_mass_lower, radius=einstein_radius, bins=bins
    )
    mass_to_light_ratio_upper = instance.normalization_from_mass_angular_and_radius(
        mass_angular=einstein_mass_upper, radius=einstein_radius, bins=bins
    )

    model.mass_to_light_ratio = af.LogUniformPrior(
        lower_limit=mass_to_light_ratio_lower, upper_limit=mass_to_light_ratio_upper
    )

    return model


def mass__from_result(
    mass, result: af.Result, unfix_mass_centre: bool = False
) -> af.Model:
    """
    Returns an updated mass `Model` whose priors are initialized from previous results in a pipeline.

    It includes an option to unfix the input `mass_centre` used in the SOURCE PIPELINE, such that if the `mass_centre`
    were fixed (e.g. to (0.0", 0.0")) it becomes a free parameter in this pipeline.

    This function generically links any `MassProfile` to any `MassProfile`, pairing parameters which share the
    same path.

    Parameters
    ----------
    results
        The result of a previous SOURCE PARAMETRIC PIPELINE or SOURCE INVERSION PIPELINE.
    unfix_mass_centre
        If the `mass_centre` was fixed to an input value in a previous pipeline, then `True` will unfix it and make it
        free parameters that are fitted for.

    Returns
    -------
    af.Model(mp.MassProfile)
        The total mass profile whose priors are initialized from a previous result.
    """

    mass.take_attributes(source=result.model.galaxies.lens.mass)

    if unfix_mass_centre and isinstance(mass.centre, tuple):

        centre_tuple = mass.centre

        mass.centre = af.Model(mass.cls).centre

        mass.centre.centre_0 = af.GaussianPrior(mean=centre_tuple[0], sigma=0.05)
        mass.centre.centre_1 = af.GaussianPrior(mean=centre_tuple[1], sigma=0.05)

    return mass


def source__from_result(
    result: af.Result, setup_hyper: al.SetupHyper, source_is_model: bool = False
) -> af.Model:
    """
    Setup the source model using the previous pipeline and search results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source can be returned as an `instance` or `model`, depending on the optional input. The default SLaM
    pipelines return parametric sources as a model (give they must be updated to properly compute a new mass
    model) and return inversions as an instance (as they have sufficient flexibility to typically not required
    updating). They use the *source_from_pevious_pipeline* method of the SLaM class to do this.

    Parameters
    ----------
    result : af.Result
        The result of the previous source pipeline.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    source_is_model : bool
        If `True` the source is returned as a *model* where the parameters are fitted for using priors of the
        search result it is loaded from. If `False`, it is an instance of that search's result.
    """

    hyper_galaxy = setup_hyper.hyper_galaxy_source_from_result(result=result)

    if result.instance.galaxies.source.pixelization is None:

        if source_is_model:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.model.galaxies.source.bulge,
                disk=result.model.galaxies.source.disk,
                envelope=result.model.galaxies.source.envelope,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                bulge=result.instance.galaxies.source.bulge,
                disk=result.instance.galaxies.source.disk,
                envelope=result.instance.galaxies.source.envelope,
                hyper_galaxy=hyper_galaxy,
            )

    if hasattr(result, "hyper"):

        if source_is_model:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.hyper.instance.galaxies.source.pixelization,
                regularization=result.hyper.model.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.hyper.instance.galaxies.source.pixelization,
                regularization=result.hyper.instance.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

    else:

        if source_is_model:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.instance.galaxies.source.pixelization,
                regularization=result.model.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return af.Model(
                al.Galaxy,
                redshift=result.instance.galaxies.source.redshift,
                pixelization=result.instance.galaxies.source.pixelization,
                regularization=result.instance.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )


def source__from_result_model_if_parametric(
    result: af.Result, setup_hyper: al.SetupHyper
) -> af.Model:
    """
    Setup the source model for a MASS PIPELINE using the previous SOURCE PIPELINE results.

    The source light model is not specified by the  MASS PIPELINE and the previous SOURCE PIPELINE is used to
    determine whether the source model is parametric or an inversion.

    The source is returned as a model if it is parametric (given its parameters must be fitted for to properly compute
    a new mass model) whereas inversions are returned as an instance (as they have sufficient flexibility to not
    require updating). This behaviour can be customized in SLaM pipelines by replacing this method with the
    `source__from_result` method.

    Parameters
    ----------
    result
        The result of the previous source pipeline.
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    """
    if result.instance.galaxies.source.pixelization is None:
        return source__from_result(
            result=result, setup_hyper=setup_hyper, source_is_model=True
        )
    return source__from_result(
        result=result, setup_hyper=setup_hyper, source_is_model=False
    )
