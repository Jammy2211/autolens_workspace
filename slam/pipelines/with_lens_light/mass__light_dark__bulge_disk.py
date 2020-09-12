import autofit as af
import autolens as al

"""
This pipeline fits the mass of a strong lens using an image with the lens light included, using a decomposed 
_LightMassProfile__, dark _MassProfile_ profile. 

The lens light is the bulge-disk model chosen by the previous Light pipeline and source model chosen by the Source 
pipeline. The mass model is initialized using results from these pipelines.

The pipeline is two phases:

Phase 1:

    Fit the lens light and mass model as a decomposed profile, using the lens light and source model from 
    a previous pipeline. The lens _LightProfile_'s are fixed to the results of the previous pipeline to provide a fast
    initialization of the new _MassProfile_ parameters.
    
    Lens Light & Mass: Depends on previous Light pipeline.
    Lens Mass: _LightMassProfile_'s + SphericalNFW + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: source__sersic.py and / or source__inversion.py and light__bulge_disk.py
    Prior Passing: Lens Light (instance -> previous pipeline), Source (instance -> previous pipeline)
    Notes: Fixes the lens _LightProfile_ and Source to the results of the previous pipeline.

Phase 2:
    
    Include all previously fixed lens _LightProfile_ parameters in the model, initializing the _MassProflie_ parameters
    from the results of phase 1.
    
    Lens Light & Mass: Depends on previous Light pipeline.
    Lens Mass: _LightMassProfile_'s + SphericalNFW + ExternalShear
    Source Light: Previous Pipeline Source.
    Previous Pipelines: None
    Prior Passing: Lens Light & Mass (model -> phase 1), source (model / instance -> previous pipeline)
    Notes: If the source is parametric, its parameters are varied, if its an _Inversion_, they are fixed.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__light_dark"

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
        slam.mass_tag,
    ]

    """SLaM: Set whether shear is Included in the mass model."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline(index=-1)

    """
    Phase 1: Fit the lens galaxy's light and mass and one source galaxy, where we:

        1) Fix the lens galaxy's light using the the _LightProfile_'s inferred in the previous 'light' pipeline, 
           including assumptions related to the geometric alignment of different components.
        2) Pass priors on the lens galaxy's _SphericalNFW_ _MassProfile_'s centre using the EllipticalIsothermal fit 
           of the previous pipeline, if the NFW centre is a free parameter.
        3) Pass priors on the lens galaxy's shear using the _ExternalShear_ fit of the previous pipeline.
        4) Pass priors on the source galaxy's light using the fit of the previous pipeline.
    """

    """SLaM: Set if the Sersic bulge is modeled with or without a mass-to-light gradient."""

    bulge = slam.pipeline_mass.bulge_light_and_mass_prior_model

    """SLaM: Set if the disk is modeled as an _EllipticalSersic_ or _EllipticalExponential_ with or without a mass-to-light gradient."""

    disk = slam.pipeline_mass.disk_light_and_mass_prior_model

    """SLaM: Set all the mass-to-light ratios of all light and mass profiles to the same value, if set as constant."""

    slam.pipeline_mass.set_mass_to_light_ratios_of_light_and_mass_prior_models(
        light_and_mass_prior_models=[bulge, disk]
    )

    """SLaM: Include a Super-Massive Black Hole (SMBH) in the mass model is specified in _SLaMPipelineMass_."""

    smbh = slam.pipeline_mass.smbh_from_centre(
        centre=af.last.instance.galaxies.lens.bulge.centre
    )

    bulge.centre = af.last.instance.galaxies.lens.bulge.centre
    bulge.elliptical_comps = af.last.instance.galaxies.lens.bulge.elliptical_comps
    bulge.intensity = af.last.instance.galaxies.lens.bulge.intensity
    bulge.effective_radius = af.last.instance.galaxies.lens.bulge.effective_radius
    bulge.sersic_index = af.last.instance.galaxies.lens.bulge.sersic_index

    disk.centre = af.last.instance.galaxies.lens.disk.centre
    disk.elliptical_comps = af.last.instance.galaxies.lens.disk.elliptical_comps
    disk.phi = af.last.instance.galaxies.lens.disk.phi
    disk.intensity = af.last.instance.galaxies.lens.disk.intensity
    disk.effective_radius = af.last.instance.galaxies.lens.disk.effective_radius

    if slam.pipeline_light.disk_as_sersic:
        disk.sersic_index = af.last.instance.galaxies.lens.disk.sersic_index

    dark = af.PriorModel(al.mp.SphericalNFWMCRLudlow)

    if slam.pipeline_mass.align_bulge_dark_centre:
        dark.centre = bulge.centre
    else:
        dark.centre = af.last.model.galaxies.lens.bulge.centre

    dark.mass_at_200 = af.LogUniformPrior(lower_limit=5e8, upper_limit=5e14)

    dark.redshift_object = slam.redshift_lens
    dark.redshift_source = slam.redshift_source

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
        dark=dark,
        shear=shear,
        smbh=smbh,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    source = slam.source_from_previous_pipeline(index=0, source_is_model=False)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_bulge_disk__mass_mlr_dark__source__fixed_lens_light",
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

    bulge = slam.pipeline_mass.bulge_light_and_mass_prior_model
    disk = slam.pipeline_mass.disk_light_and_mass_prior_model

    bulge.centre = af.last[-1].model.galaxies.lens.bulge.centre
    bulge.elliptical_comps = af.last[-1].model.galaxies.lens.bulge.elliptical_comps
    bulge.intensity = af.last[-1].model.galaxies.lens.bulge.intensity
    bulge.effective_radius = af.last[-1].model.galaxies.lens.bulge.effective_radius
    bulge.sersic_index = af.last[-1].model.galaxies.lens.bulge.sersic_index

    bulge.mass_to_light_ratio = (
        phase1.result.model.galaxies.lens.bulge.mass_to_light_ratio
    )

    disk.centre = af.last[-1].model.galaxies.lens.disk.centre
    disk.elliptical_comps = af.last[-1].model.galaxies.lens.disk.elliptical_comps
    disk.intensity = af.last[-1].model.galaxies.lens.disk.intensity
    disk.effective_radius = af.last[-1].model.galaxies.lens.disk.effective_radius
    if slam.pipeline_light.disk_as_sersic:
        disk.sersic_index = af.last[-1].model.galaxies.lens.disk.sersic_index

    disk.mass_to_light_ratio = (
        phase1.result.model.galaxies.lens.disk.mass_to_light_ratio
    )

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
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
