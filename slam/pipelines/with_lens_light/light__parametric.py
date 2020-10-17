import autofit as af
import autolens as al

"""
This pipeline fits the lens light of a strong lens using a two component bulge + disk model.

The mass model and source are initialized using an already run `source` pipeline. Although the lens light was
fitted in this pipeline, we do not use this model to set priors in this pipeline.

The bulge and disk are modeled using `EllipticalSersic` and EllipticalExponential profiles respectively. Their alignment
(centre, elliptical components) and whether the disk component is instead modeled using an `EllipticalSersic` profile
can be customized using the pipeline slam.

The pipeline is one phase:

Phase 1:

    Fit the lens light using a bulge + disk model, with the lens mass and source fixed to the
    result of the previous pipeline
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: Previous `source` pipeline.
    Source Light: Previous `source` pipeline.
    Previous Pipelines: source__parametric.py and / or source__inversion.py
    Prior Passing: Lens Mass (instance -> previous pipeline), Source (instance -> previous pipeliine).
    Notes: Can be customized to vary the lens mass and source.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_light[parametric]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The bulge model, disk model, etc. and whether their centres / elliptical_comps are aligned.
        3) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = f"{slam.path_prefix}/{pipeline_name}/{slam.source_tag}/{slam.light_parametric_tag}"

    """
    Phase 1: Fit the lens `Galaxy`'s light, where we:

        1) Fix the lens `Galaxy`'s mass and source galaxy to the results of the previous pipeline.
        2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.
    """

    """SlaM:  If hyper-galaxy noise scaling is on, it may over-scale the noise making this new `LightProfile` 
    fit the data less well. This can be circumvented by including the noise scaling as a free parameter."""

    hyper_galaxy = slam.setup_hyper.hyper_galaxy_lens_from_previous_pipeline(
        noise_factor_is_model=True
    )

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=slam.pipeline_light.setup_light.bulge_prior_model,
        disk=slam.pipeline_light.setup_light.disk_prior_model,
        envelope=slam.pipeline_light.setup_light.envelope_prior_model,
        mass=af.last.instance.galaxies.lens.mass,
        shear=af.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    """SLaM: Use the Source pipeline source as an instance (whether its parametric or an Inversion)."""

    source = slam.source_from_previous_pipeline(source_is_model=False)

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_light[parametric]_mass[fixed]_source[fixed]",
            n_live_points=100,
        ),
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
    )

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, path_prefix, phase1)
