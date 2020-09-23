import autofit as af
import autolens as al

"""
This pipeline fits the lens light of a strong lens using a single `EllipticalSersic` model.

The mass model and source are initialized using an already run `source` pipeline. Although the lens light was
fitted in this pipeline, we do not use this model to set priors in this pipeline.

The pipeline is one phase:

Phase 1:

    Fit the lens light using an `EllipticalSersic` model, with the lens mass and source fixed to the result of the 
    previous pipeline.
    
    Lens Light & Mass: EllipticalSersic
    Lens Mass: Previous `source` pipeline.
    Source Light: Previous `source` pipeline.
    Previous Pipelines: source__sersic.py and / or source__inversion.py
    Prior Passing: Lens Mass (instance -> previous pipeline), Source (instance -> previous pipeliine).
    Notes: Can be customized to vary the lens mass and source.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_light__sersic"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """

    folders = slam.folders + [pipeline_name, slam.source_tag, slam.light_tag]

    """
    Phase 1: Fit the lens galaxy`s light, where we:

        1) Fix the lens galaxy`s mass and source galaxy to the results of the previous pipeline.
        2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.
    """

    """SlaM:  If hyper-galaxy noise scaling is on, it may over-scale the noise making this new `LightProfile` 
    fit the data less well. This can be circumvented by including the noise scaling as a free parameter."""

    hyper_galaxy = slam.setup_hyper.hyper_galaxy_lens_from_previous_pipeline(
        noise_factor_is_model=True
    )

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        sersic=al.lp.EllipticalSersic,
        mass=af.last.instance.galaxies.lens.mass,
        shear=af.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    """SLaM: Use the Source pipeline source as an instance (whether its parametric or an Inversion)."""

    source = slam.source_from_previous_pipeline(source_is_model=False)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_sersic__mass__source",
        folders=folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=60),
    )

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1)
