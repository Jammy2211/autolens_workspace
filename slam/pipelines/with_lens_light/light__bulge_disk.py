import autofit as af
import autolens as al

"""
This pipeline fits the lens light of a strong lens using a two component bulge + disk model.

The mass model and source are initialized using an already run 'source' pipeline. Although the lens light was
fitted in this pipeline, we do not use this model to set priors in this pipeline.

The bulge and disk are modeled using _EllipticalSersic_ and EllipticalExponential profiles respectively. Their alignment
(centre, elliptical components) and whether the disk component is instead modeled using an _EllipticalSersic_ profile
can be customized using the pipeline slam.

The pipeline is one phase:

Phase 1:

    Fit the lens light using a bulge + disk model, with the lens mass and source fixed to the
    result of the previous pipeline
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: Previous 'source' pipeline.
    Source Light: Previous 'source' pipeline.
    Previous Pipelines: source__sersic.py and / or source__inversion.py
    Prior Passing: Lens Mass (instance -> previous pipeline), Source (instance -> previous pipeliine).
    Notes: Can be customized to vary the lens mass and source.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_light__bulge_disk"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The bulge + disk centres or elliptical_comps are aligned.
        3) The disk component of the lens light model is an _EllipticalExponential_ or _EllipticalSersic_ profile.
        4) The lens galaxy mass model includes an  _ExternalShear_.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.setup_hyper.tag,
        slam.source_tag,
        slam.light_tag,
    ]

    """
    Phase 1: Fit the lens galaxy's light, where we:

        1) Fix the lens galaxy's mass and source galaxy to the results of the previous pipeline.
        2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.
    """

    """SlaM:  If hyper-galaxy noise scaling is on, it may over-scale the noise making this new _LightProfile_ 
    fit the data less well. This can be circumvented by including the noise scaling as a free parameter."""

    hyper_galaxy = slam.setup_hyper.hyper_galaxy_lens_from_previous_pipeline(
        noise_factor_is_model=True
    )

    """SLaM: Set whether the disk is modeled as an _EllipticalSersic_ or _EllipticalExponential_."""

    disk = slam.pipeline_light.setup_light.disk_prior_model

    bulge = af.PriorModel(al.lp.EllipticalSersic)

    """SLaM: Set the alignment of the bulge and disk's centres and elliptical components."""

    slam.pipeline_light.setup_light.align_bulge_and_disk_centre_and_elliptical_comps(
        bulge_prior_model=bulge, disk_prior_model=disk
    )

    """SLaM: include a thid component, the envelope, in the model if included in the _LightSetup_."""

    envelope = slam.pipeline_light.setup_light.envelope_prior_model

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
        envelope=envelope,
        mass=af.last.instance.galaxies.lens.mass,
        shear=af.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    """SLaM: Use the Source pipeline source as an instance (whether its parametric or an Inversion)."""

    source = slam.source_from_previous_pipeline(source_is_model=False)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_bulge_disk__mass__source",
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
