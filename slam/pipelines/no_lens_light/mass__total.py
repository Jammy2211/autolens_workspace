import autofit as af
import autolens as al

"""
In this pipeline, we fit the mass of a strong lens using an input _MassProfile_ (default=_EllipticalPowerLaw_) + 
shear model.

The source model is chosen and mass model is initialized using the previously run Source pipeline.

The pipeline is one phases:

Phase 1:

    Fit the lens mass model as a power-law, using the source model from a previous pipeline.
    Lens Mass: MassProfile + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipeline: no_lens_light/source/*/mass_sie__source_*py
    Prior Passing: Lens Mass (model -> previous pipeline), source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an _Inversion_, they are fixed.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__total"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  _ExternalShear_.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.setup_hyper.tag,
        slam.source_tag,
        slam.mass_tag,
    ]

    """SLaM: Set whether shear is included in the mass model."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline(index=0)

    """
    Phase 1: Fit the lens's _MassProfile_'s and source, where we:

        1) Use the source galaxy of the 'source' pipeline.
        2) Set priors on the lens galaxy _MassProfile_'s using the EllipticalIsothermal and ExternalShear of previous pipelines.
    """

    """Setup the _EllipticalPowerLaw_ _MassProfile_ and initialize its priors from the _EllipticalIsothermal_."""

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = af.last.model.galaxies.lens.mass.centre
    mass.elliptical_comps = af.last.model.galaxies.lens.mass.elliptical_comps
    mass.einstein_radius = af.last.model.galaxies.lens.mass.einstein_radius

    """
    SLaM: Setup the source model, which uses a variable parametric profile or fixed _Inversion_ model.
    """

    source = slam.source_from_source_pipeline_for_mass_pipeline(index=0)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_power_law__source",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=slam.redshift_lens, mass=mass, shear=shear),
            source=source,
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase1 = phase1.extend_with_stochastic_phase(
        stochastic_method="gaussian",
        stochastic_sigma=0.0,
        stochastic_search=af.DynestyStatic(n_live_points=100),
    )

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1)
