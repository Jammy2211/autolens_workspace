import autofit as af
import autolens as al

from typing import Union, Dict, Optional


def hyper_fit(
    setup_hyper: al.SetupHyper,
    result: af.Result,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    include_hyper_image_sky: bool = False,
):
    """
    Perform a hyper-fit, which extends a model-fit with an additional fit which fixes the non-hyper components of the
    model (e.g., `LightProfile`'s, `MassProfile`) to the `Result`'s maximum likelihood fit. The hyper-fit then treats
    only the hyper-model components as free parameters, which are any of the following model components:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
    data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `Inversion` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_hyper
        The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper model.
    analysis
        An analysis which is used to fit imaging or interferometer data with a model.
    include_hyper_image_sky
        Whether to include the hyper image sky component, irrespective of the `setup_hyper`.

    Returns
    -------
    af.Result
        The result of the hyper model-fit, which has a new attribute `result.hyper` that contains updated parameter
        values for the hyper-model components for passing to later model-fits.
    """

    hyper_model = al.util.model.hyper_model_from(
        setup_hyper=setup_hyper,
        result=result,
        include_hyper_image_sky=include_hyper_image_sky,
    )

    return al.util.model.hyper_fit(
        hyper_model=hyper_model,
        setup_hyper=setup_hyper,
        result=result,
        analysis=analysis.no_positions,
    )


def stochastic_fit(
    result: af.Result,
    analysis: Union[al.AnalysisImaging, al.AnalysisInterferometer],
    include_lens_light: bool = False,
    include_pixelization: bool = False,
    include_regularization: bool = False,
    search_cls: af.NonLinearSearch = af.DynestyStatic,
    search_dict: Optional[Dict] = None,
):
    """
    Extend a model-fit with a stochastic model-fit, which refits a model but introduces a log likelihood cap whereby
    all model-samples with a likelihood above this cap are rounded down to the value of the cap.

    This `log_likelihood_cap` is determined by sampling ~250 log likelihood values from the original model's maximum
    log likelihood model. However, the pixelization used to reconstruct the source of each model evaluation uses a
    different KMeans seed, such that each reconstruction uses a unique pixel-grid. The model must therefore use a
    pixelization which uses the KMeans method to construct the pixel-grid, for example the `VoronoiBrightnessImage`.

    The cap is computed as the mean of these ~250 values and it is introduced to avoid underestimated errors due
    to artificial likelihood boosts.

    Parameters
    ----------
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the hyper
        model.
    analysis
        An analysis which is used to fit imaging or interferometer data with a model.
    include_lens_light
        If the lens light is included as a model component in the model with free parameters that are fitted for (if
        `False` it is passed as an `instance`).
    include_pixelization
        If the source pixelization is included as a model component in the model with free parameters that are fitted
        for (if `False` it is passed as an `instance`).
    include_regularization
        If the source regularization is included as a model component in the model with free parameters that are
        fitted for (if `False` it is passed as an `instance`).
    """

    if search_dict is None:
        search_dict = {"nlive": 100}

    stochastic_model = al.util.model.stochastic_model_from(
        result=result,
        include_lens_light=include_lens_light,
        include_pixelization=include_pixelization,
        include_regularization=include_regularization,
    )

    return al.util.model.stochastic_fit(
        stochastic_model=stochastic_model,
        search_cls=search_cls,
        search_dict=search_dict,
        result=result,
        analysis=analysis,
    )
