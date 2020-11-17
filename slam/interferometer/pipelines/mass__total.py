from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit the mass of a strong lens using an input `MassProfile` (default=`EllipticalPowerLaw`) + 
shear model.

The source model is chosen and mass model is initialized using the previously run Source pipeline.

The pipeline uses one phase:

Phase 1:

    Fit the lens mass model as a power-law, using the source model from a previous pipeline.
    Lens Mass: MassProfile (default=EllipticalPowerLaw) + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipeline: no_lens_light/source/*/mass_sie__source_*py
    Prior Passing: Lens Mass (model -> previous pipeline), source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an `Inversion`, they are fixed.
"""


def make_pipeline(slam, settings, real_space_mask):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass[total]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = path.join(
        slam.path_prefix, pipeline_name, slam.source_tag, slam.mass_tag
    )

    """SLaM: Set whether shear is included in the mass model."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline(index=0)

    """
    Phase 1: Fit the lens`s `MassProfile`'s and source, where we:

        1) Set priors on the lens galaxy `MassProfile`'s using the `EllipticalIsothermal` and `ExternalShear` 
           of previous pipelines.
        2) Use the source galaxy model of the `source` pipeline.
        3) Fit this source as a model if it is parametric and as an instance if it is an `Inversion`.
    """

    """Setup the `MassProfile`.and initialize its priors from the `EllipticalIsothermal`."""

    mass = slam.pipeline_mass.setup_mass.mass_prior_model_with_updated_priors(
        index=0, unfix_mass_centre=True
    )

    """
    SLaM: Setup the source model, which uses a variable parametric profile or fixed `Inversion` model.
    """

    source = slam.source_from_previous_pipeline_model_if_parametric(index=0)

    phase1 = al.PhaseInterferometer(
        search=af.DynestyStatic(name="phase[1]_mass[total]_source", n_live_points=100),
        galaxies=dict(
            lens=al.GalaxyModel(redshift=slam.redshift_lens, mass=mass, shear=shear),
            source=source,
        ),
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        real_space_mask=real_space_mask,
    )

    # phase1 = phase1.extend_with_stochastic_phase(
    #     stochastic_method="gaussian",
    #     stochastic_sigma=0.0,
    #     stochastic_search=af.DynestyStatic(n_live_points=100),
    # )

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, path_prefix, phase1)
