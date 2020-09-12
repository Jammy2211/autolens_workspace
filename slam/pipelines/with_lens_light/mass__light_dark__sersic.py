import autofit as af
import autolens as al

"""
This pipeline fits the mass of a strong lens using an image with the lens light included, using a decomposed 
_LightMassProfile__, dark _MassProfile_ profile. 

The lens light is the _EllipticalSersic_ model chosen by the previous Light pipeline and source model chosen by the 
Source pipeline. The mass model is initialized using results from these pipelines.

The pipeline is two phases:

Phase 1:

    Fit the lens light and mass model as a decomposed profile, using the lens light and source model from 
    a previous pipeline. The lens _LightProfile_'s are fixed to the results of the previous pipeline to provide a fast
    initialization of the new _MassProfile_ parameters.
    
    Lens Light & Mass: EllipticalSersic
    Lens Mass: LightMassProfile's + SphericalNFW + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: source__sersic.py and / or source__inversion.py and light__sersic.py
    Prior Passing: Lens Light (instance -> previous pipeline), Source (instance -> previous pipeline)
    Notes: Fixes the lens _LightProfile_ and Source to the results of the previous pipeline.

Phase 2:
    
    Include all previously fixed lens _LightProfile_ parameters in the model, initializing the _MassProflie_ parameters
    from the results of phase 1.
    
    Lens Light & Mass: EllipticalSersic
    Lens Mass: _LightMassProfile_'s + SphericalNFW + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: None
    Prior Passing: Lens Light & Mass (model -> phase 1), source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an _Inversion_, they are fixed.
"""


def make_pipeline(slam, settings):
    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__light_dark__sersic"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  _ExternalShear_.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.setup_hyper.tag,
        slam.source_tag,
        slam.light_tag,
        slam.mass_tag,
    ]

    """SLaM: Set whether shear is Included in the mass model."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline(index=-1)

    """
    Phase 1: Fit the lens galaxy's light and mass and one source galaxy, where we:

        1) Fix the lens galaxy's light using the the _LightProfile_ inferred in the previous 'light' pipeline.
        2) Pass priors on the lens galaxy's SphericalNFW _MassProfile_'s centre using the EllipticalIsothermal fit of the
           previous pipeline, if the NFW centre is a free parameter.
        3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
        4) Pass priors on the source galaxy's light using the _EllipticalSersic_ of the previous pipeline.
    """

    """SLaM: Set if the _EllipticalSersic_ is modeled with or without a mass-to-light gradient."""

    sersic = slam.pipeline_mass.bulge_light_and_mass_prior_model

    """SLaM: Include a Super-Massive Black Hole (SMBH) in the mass model is specified in _SLaMPipelineMass_."""

    smbh = slam.pipeline_mass.smbh_from_centre(
        centre=af.last.instance.galaxies.lens.sersic.centre
    )

    sersic.centre = af.last.instance.galaxies.lens.sersic.centre
    sersic.elliptical_comps = af.last.instance.galaxies.lens.sersic.elliptical_comps
    sersic.intensity = af.last.instance.galaxies.lens.sersic.intensity
    sersic.effective_radius = af.last.instance.galaxies.lens.sersic.effective_radius
    sersic.sersic_index = af.last.instance.galaxies.lens.sersic.sersic_index

    """SLaM: Align the centre of the _LightProfile_ and dark matter _MassPrfile_ if input in _SetupMassLightDark_."""

    dark = af.PriorModel(al.mp.SphericalNFWMCRLudlow)

    if slam.pipeline_mass.align_bulge_dark_centre:
        dark.centre = sersic.centre
    else:
        dark.centre = af.last.model.galaxies.lens.sersic.centre

    dark.mass_at_200 = af.LogUniformPrior(lower_limit=5e8, upper_limit=5e14)

    dark.redshift_object = slam.redshift_lens
    dark.redshift_source = slam.redshift_source

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        sersic=sersic,
        dark=dark,
        shear=shear,
        smbh=smbh,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    source = slam.source_from_previous_pipeline(index=0, source_is_model=False)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_sersic__mass_mlr_dark__source__fixed_lens_light",
        folders=folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=30),
    )

    """
    Phase 2: Fit the lens galaxy's light and mass and source galaxy using the results of phase 1 as
    initialization
    """

    sersic = slam.pipeline_mass.sersic_light_and_mass_prior_model

    sersic.centre = af.last[-1].model.galaxies.lens.sersic.centre
    sersic.elliptical_comps = af.last[-1].model.galaxies.lens.sersic.elliptical_comps
    sersic.intensity = af.last[-1].model.galaxies.lens.sersic.intensity
    sersic.effective_radius = af.last[-1].model.galaxies.lens.sersic.effective_radius
    lens.sersic.sersic_index = af.last[-1].model.galaxies.lens.sersic.sersic_index

    sersic.mass_to_light_ratio = (
        phase1.result.model.galaxies.lens.sersic.mass_to_light_ratio
    )

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        sersic=sersic,
        dark=phase1.result.model.galaxies.lens.dark,
        shear=af.last[-2].model.galaxies.lens.shear,
        smbh=phase1.result.model.galaxies.lens.smbh,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2__light_bulge_disk__mass_mlr_dark__source",
        folders=folders,
        galaxies=dict(lens=lens, source=phase1.result.model.galaxies.source),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    if not slam.setup_hyper.hyper_fixed_after_source:
        phase2 = phase2.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
