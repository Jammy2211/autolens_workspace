import autofit as af
import autolens as al

"""
In this pipeline, we fit the mass of a strong lens using a power-law + shear model.

The lens light, mass model and source are initialized using already run 'source' and 'light' pipelines.

The pipeline is one phases:

Phase 1:

Fit the lens mass model as a power-law, using the source model from a previous pipeline.
Lens Mass: Light + EllipticalPowerLaw + ExternalShear
Source Light: Previous Pipeline Source.
Previous Pipelines: no_lens_light/source/*/lens_sie__source_*py
Prior Passing: Lens Mass (model -> previous pipeline), source (model / instance -> previous pipeline)
Notes: If the source is parametric, its parameters are varied, if its an inversion, they are fixed.
"""


def make_pipeline(
    slam, settings, phase_folders=None, redshift_lens=0.5, redshift_source=1.0
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_stochastic"

    """
    This pipeline is tagged according to whether:

    1) Hyper-fitting settings (galaxies, sky, background noise) are used.
    2) The lens galaxy mass model includes an external shear.
    3) The lens's light model is fixed or variable.
    """

    phase_folders.append(pipeline_name)
    phase_folders.append(slam.hyper.tag)
    phase_folders.append(slam.source.tag)
    phase_folders.append(slam.mass.tag)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_power_law__source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=af.last.model.galaxies.lens, source=af.last.model.galaxies.source
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=75, sampling_efficiency=0.2, evidence_tolerance=0.8
        ),
    )

    return al.PipelineDataset(pipeline_name, phase1)
