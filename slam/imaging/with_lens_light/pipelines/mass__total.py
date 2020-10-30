import autofit as af
import autolens as al

"""
This pipeline fits the mass of a strong lens using a _MassProfile (default=`EllipticalPowerLaw`) representing total 
mass distribution (e.g. stellar and dark) of the lens `Galaxy`.

The lens light and source models use those chosen in the already run `source` and `light` pipelines, and the mass model
is initialized using results from these pipelines.

The pipeline is one phase:

Phase 1:

    Fit the lens mass model _MassProfile (default=`EllipticalPowerLaw`), using the lens light and source model from a 
    previous pipeline.
    
    Lens Light: Previous Light Pipeline Light.
    Lens Mass: Light + EllipticalPowerLaw + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: source__parametric.py and / or source__inversion.py and light__parametric.py or light__parametric.py
    Prior Passing: Lens Light (instance -> previous piepline) Lens Mass (model -> previous pipeline), 
                   Source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an `Inversion`, they are fixed.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass[total]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
        3) The lens`s light model is fixed or variable.
    """

    path_prefix = f"{slam.path_prefix}/{pipeline_name}/{slam.source_tag}/{slam.light_parametric_tag}/{slam.mass_tag}"

    """SLaM: Set whether shear is included in the mass model using the `ExternalShear` model of the Source pipeline."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline(index=-1)

    """
    Phase 1: Fit the lens `Galaxy`'s light and mass and one source galaxy, where we:

        1) Use the source galaxy of the `source` pipeline.
        2) Use the lens galaxy light of the `light` pipeline.
        3) Set priors on the lens galaxy `MassProfile`'s using the `EllipticalIsothermal` and `ExternalShear` of 
           previous pipelines.
    """

    mass = slam.pipeline_mass.setup_mass.mass_prior_model_with_updated_priors(
        index=-1, unfix_mass_centre=True
    )

    """SLaM: Use the source and lens light models from the previous *Source* and *Light* pipelines."""

    lens = slam.lens_from_light_parametric_pipeline_for_mass_total_pipeline(
        mass=mass, shear=shear
    )

    source = slam.source_from_previous_pipeline_model_if_parametric()

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_light[parametric]_mass[total]_source", n_live_points=100
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
