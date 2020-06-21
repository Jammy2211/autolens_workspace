import autofit as af
import autolens as al

"""
In this pipeline, we fit the mass of a strong lens using an SIE + shear model.

The mass model and source are initialized using an already run 'source' pipeline.

The pipeline is one phases:

Phase 1:

    Fit the lens mass model as a power-law, using the source model from a previous pipeline.
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipeline: no_lens_light/source/*/lens_sie__source_*py
    Prior Passing: Lens Mass (model -> previous pipeline), source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an inversion, they are fixed.
"""


def make_pipeline(slam, settings, folders=None, redshift_lens=0.5, redshift_source=1.0):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__sie"

    """TAG: Setup the lens mass tag for pipeline tagging"""
    slam.set_mass_type(mass_type="sie")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an external shear.
    """

    slam.folders.append(pipeline_name)
    slam.folders.append(slam.hyper.tag)
    slam.folders.append(slam.source.tag)
    slam.folders.append(slam.mass.tag)

    """SLaM: Set whether shear is Included in the mass model."""

    shear = slam.mass.shear_from_previous_pipeline

    """
    Phase 1: Fit the lens galaxy's mass and source, where we:

        1) Use the source galaxy of the 'source' pipeline.
        2) Set priors on the lens galaxy mass using the EllipticalIsothermal and ExternalShear of previous pipelines.
    """

    """
    SLaM: Setup the source model, which uses a variable parametric profile or fixed inversion model.
    """

    source = slam.source_from_previous_pipeline()

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sie__source",
        folders=slam.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=af.last.model.galaxies.lens.mass,
                shear=shear,
            ),
            source=source,
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=75, sampling_efficiency=0.2, evidence_tolerance=0.8
        ),
    )

    if not slam.hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup=slam.hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1)
