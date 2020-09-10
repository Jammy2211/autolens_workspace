import autofit as af
import autolens as al

"""
In this pipeline, we fit the mass of a strong lens using a _EllipticalPowerLaw_ + shear model.

The lens light, mass model and source are initialized using already run 'source' and 'light' pipelines.

The pipeline is one phases:

Phase 1:

    Fit the lens mass model as a power-law, using the source model from a previous pipeline.
    Lens Mass: Light + EllipticalPowerLaw + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: no_lens_light/source/*/mass_sie__source_*py
    Prior Passing: Lens Mass (model -> previous pipeline), source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an _Inversion_, they are fixed.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__power_law"

    """TAG: Setup the lens mass tag for pipeline tagging"""
    slam.set_mass_type(mass_type="power_law")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  _ExternalShear_.
        3) The lens's light model is fixed or variable.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.setup_hyper.tag,
        slam.setup_source.tag,
        slam.setup_light.tag,
        slam.pipeline_mass.tag,
    ]

    """SLaM: Set whether shear is Included in the mass model."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline

    """
    Phase 1: Fit the lens galaxy's light and mass and one source galaxy, where we:

        1) Use the source galaxy of the 'source' pipeline.
        2) Use the lens galaxy light of the 'light' pipeline.
        3) Set priors on the lens galaxy _MassProfile_'s using the EllipticalIsothermal and ExternalShear of previous pipelines.
    """

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)
    mass.centre = af.last[-1].model.galaxies.lens.mass.centre
    mass.elliptical_comps = af.last[-1].model.galaxies.lens.mass.elliptical_comps
    mass.einstein_radius = af.last[-1].model.galaxies.lens.mass.einstein_radius

    """SLaM: Use the source and lens light models from the previous *Source* and *Light* pipelines."""

    lens = slam.lens_from_light_pipeline_for_mass_pipeline(
        redshift_lens=slam.redshift_lens, mass=mass, shear=shear
    )
    source = slam.source_from_previous_pipeline_model_if_parametric()

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_power_law__source",
        folders=folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1)
