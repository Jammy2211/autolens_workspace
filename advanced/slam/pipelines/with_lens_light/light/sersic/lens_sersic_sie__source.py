import autofit as af
import autolens as al

"""
In this pipeline, we fit the lens light of a strong lens using a single Sersic model.

The mass model and source are initialized using an already run 'source' pipeline. Although the lens light was
fitted in this pipeline, we do not use this model to set priors in this pipeline.

The pipeline is one phase:

Phase 1:

    Fit the lens light using a Sersic model, with the lens mass and source fixed to the result of the previous pipeline.
    
    Lens Light & Mass: EllipticalSersic
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: Previous 'source' pipeline.
    Previous Pipelines: with_lens_light/source/*/lens_bulge_disk_sie__source_*.py
    Prior Passing: Lens Mass (instance -> previous pipeline), Source (instance -> previous pipeliine).
    Notes: Can be customized to vary the lens mass and source.
"""


def make_pipeline(slam, settings, redshift_lens=0.5):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_light__sersic"

    """TAG: Setup the lens light tag for pipeline tagging"""
    slam.set_light_type(light_type="sersic")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an external shear.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.hyper.tag,
        slam.source.tag,
        slam.light.tag,
    ]

    """
    Phase 1: Fit the lens galaxy's light, where we:

        1) Fix the lens galaxy's mass and source galaxy to the results of the previous pipeline.
        2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.

    If hyper-galaxy noise scaling is on, it may over-scale the noise making this new _LightProfile_ fit the data less
    well. This can be circumvented by including the noise scaling as a free parameter.
    """

    if slam.hyper.hyper_galaxies:

        hyper_galaxy = af.PriorModel(al.HyperGalaxy)

        hyper_galaxy.noise_factor = (
            af.last.hyper_combined.model.galaxies.lens.hyper_galaxy.noise_factor
        )
        hyper_galaxy.contribution_factor = (
            af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = (
            af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.noise_power
        )

    else:

        hyper_galaxy = None

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lp.EllipticalSersic,
        mass=af.last.instance.galaxies.lens.mass,
        shear=af.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    """SLaM: Use the Source pipeline source as an instance (whether its parametric or an Inversion)."""

    source = slam.source_from_source_pipeline_for_light_pipeline()

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic_sie__source",
        folders=folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=60),
    )

    if not slam.hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup=slam.hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1)
