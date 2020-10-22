import autofit as af
import autolens as al

"""
This pipeline fits the mass of a strong lens using an image with the lens light included, using a decomposed 
_LightMassProfile__, dark `MassProfile` profile. 

The lens light is the bulge-disk model chosen by the previous Light pipeline and source model chosen by the Source 
pipeline. The mass model is initialized using results from these pipelines.

The pipeline is two phases:

Phase 1:

    Fit the lens light and mass model as a decomposed profile, using the lens light and source model from 
    a previous pipeline. The lens `LightProfile`'s are fixed to the results of the previous pipeline to provide a fast
    initialization of the new `MassProfile` parameters.
    
    Lens Light & Mass: Depends on previous Light pipeline.
    Lens Mass: `LightMassProfile`'s + SphericalNFW + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: source__parametric.py and / or source__inversion.py and light__parametric.py
    Prior Passing: Lens Light (instance -> previous pipeline), Source (instance -> previous pipeline)
    Notes: Fixes the lens `LightProfile` and Source to the results of the previous pipeline.

Phase 2:
    
    Include all previously fixed lens `LightProfile` parameters in the model, initializing the `MassProflie` parameters
    from the results of phase 1.
    
    Lens Light & Mass: Depends on previous Light pipeline.
    Lens Mass: `LightMassProfile`'s + SphericalNFW + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: None
    Prior Passing: Lens Light & Mass (model -> phase 1), source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an `Inversion`, they are fixed.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass[light_dark]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The bulge model, disk model, etc. and whether their centres / elliptical_comps are aligned.
        4) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = f"{slam.path_prefix}/{pipeline_name}/{slam.source_tag}/{slam.light_parametric_tag}/{slam.mass_tag}"

    """SLaM: Set whether shear is Included in the mass model."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline(index=-1)

    """
    Phase 1: Fit the lens `Galaxy`'s light and mass and one source galaxy, where we:

        1) Fix the lens `Galaxy`'s light using the the `LightProfile`'s inferred in the previous `light` pipeline, 
           including assumptions related to the geometric alignment of different components.
        2) Pass priors on the lens `Galaxy`'s `SphericalNFW` `MassProfile`'s centre using the EllipticalIsothermal fit 
           of the previous pipeline, if the NFW centre is a free parameter.
        3) Pass priors on the lens `Galaxy`'s shear using the `ExternalShear` fit of the previous pipeline.
        4) Pass priors on the source `Galaxy`'s light using the fit of the previous pipeline.
    """

    """SLaM: Fix the `LightProfile` parameters of the bulge, disk and envelope to the results of the Light pipeline."""

    bulge = slam.pipeline_mass.setup_mass.bulge_prior_instance_with_updated_priors()
    disk = slam.pipeline_mass.setup_mass.disk_prior_instance_with_updated_priors()
    envelope = (
        slam.pipeline_mass.setup_mass.envelope_prior_instance_with_updated_priors()
    )

    dark = slam.pipeline_mass.setup_mass.dark_prior_model
    dark.mass_at_200 = af.LogUniformPrior(lower_limit=5e8, upper_limit=5e14)
    dark.redshift_object = slam.redshift_lens
    dark.redshift_source = slam.redshift_source

    """SLaM: Include a Super-Massive Black Hole (SMBH) in the mass model is specified in `SLaMPipelineMass`."""

    smbh = slam.pipeline_mass.smbh_prior_model

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
        envelope=envelope,
        dark=dark,
        shear=shear,
        smbh=smbh,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    source = slam.source_from_previous_pipeline(index=0, source_is_model=False)

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_light[fixed]_mass[light_dark]_source[fixed]",
            n_live_points=30,
        ),
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
    )

    """
    Phase 2: Fit the lens `Galaxy`'s light and mass and source galaxy using the results of phase 1 as
    initialization
    """

    """SLaM: Set priors on the `LightProfile` parameters of the bulge, disk and envelope to the results of the Light pipeline."""

    bulge = slam.pipeline_mass.setup_mass.bulge_prior_model_with_updated_priors()
    disk = slam.pipeline_mass.setup_mass.disk_prior_model_with_updated_priors()
    envelope = slam.pipeline_mass.setup_mass.envelope_prior_model_with_updated_priors()

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
        envelope=envelope,
        dark=phase1.result.model.galaxies.lens.dark,
        shear=af.last[-2].model.galaxies.lens.shear,
        smbh=phase1.result.model.galaxies.lens.smbh,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_light[parametric]_mass[light_dark]_source", n_live_points=100
        ),
        galaxies=dict(lens=lens, source=phase1.result.model.galaxies.source),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
    )

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase2 = phase2.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, path_prefix, phase1, phase2)
