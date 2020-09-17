import autofit as af
import autolens as al

"""
This pipeline performs a parametric source analysis which fits a lens model (the lens's _LightProfile_ and mass) and the
source galaxy. 

This pipeline uses four phases:

Phase 1:

    Fit and subtract the lens light.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: None
    Source Light: None
    Previous Pipelines: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source _LightProfile_, using the lens subtracted image from phase 1.
    
    Lens Light: None
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: None
    Notes: Uses the lens light subtracted image from phase 1

Phase 3:

    Refit the lens _LightProfile_ using the mass model and source _LightProfile_ fixed from phase 2.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: lens mass and source (instance -> phase 2)
    Notes: None

Phase 4:

    Refine the lens _LightProfile_ and _MassProfile_ and source _LightProfile_, using priors from the previous 2 phases.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: Lens light (model -> phase 3), lens mass and source (model -> phase 2)
    Notes: None
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__sersic"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  _ExternalShear_.
    """

    folders = slam.folders + [pipeline_name, slam.source_parametric_tag]

    """
    Phase 1: Fit only the lens galaxy's light, where we:

        1) Align the bulge and disk (y,x) centre.
    """

    bulge = af.PriorModel(al.lp.EllipticalSersic)
    disk = af.PriorModel(al.lp.EllipticalExponential)

    bulge.centre = disk.centre

    """SLaM: Align the _LightProfile_ model centres (bulge and disk) with the input slam light_centre, if input."""

    bulge = slam.pipeline_source_parametric.setup_light.align_centre_to_light_centre(
        light_prior_model=bulge
    )
    disk = slam.pipeline_source_parametric.setup_light.align_centre_to_light_centre(
        light_prior_model=disk
    )

    lens = al.GalaxyModel(redshift=slam.redshift_lens, bulge=bulge, disk=disk)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_bulge_disk",
        folders=folders,
        galaxies=dict(lens=lens),
        settings=settings,
        search=af.DynestyStatic(n_live_points=75),
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(setup_hyper=slam.setup_hyper)

    """
    Phase 2: Fit the lens's _MassProfile_'s and source galaxy's _LightProfile_, where we:

        1) Fix the foreground lens _LightProfile_ to the result of phase 1.
        2) Set priors on the centre of the lens galaxy's _MassProfile_ by linking them to those inferred for
           the bulge of the _LightProfile_ in phase 1.
    """

    mass = af.PriorModel(slam.pipeline_source_parametric.setup_mass.mass_profile)

    """SLaM: Align the bulge and mass model centres if align_light_mass_centre is True."""

    if slam.pipeline_source_parametric.setup_mass.align_light_mass_centre:
        mass.centre = phase1.result.instance.galaxies.lens.bulge.centre
    else:
        mass.centre = phase1.result.model.galaxies.lens.bulge.centre

    """SLaM: Align the mass model centre with the input slam mass_centre, if input."""

    mass = slam.pipeline_source_parametric.setup_mass.align_centre_to_mass_centre(
        mass_prior_model=mass
    )

    """SLaM: The shear model is chosen below based on the input of _SetupSource_."""

    phase2 = al.PhaseImaging(
        phase_name="phase_2__mass_sie__source_seric",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=phase1.result.instance.galaxies.lens.bulge,
                disk=phase1.result.instance.galaxies.lens.disk,
                mass=mass,
                shear=slam.pipeline_source_parametric.setup_mass.shear_prior_model,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                sersic=al.lp.EllipticalSersic,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=200, walks=10),
    )

    phase2 = phase2.extend_with_multiple_hyper_phases(setup_hyper=slam.setup_hyper)

    """
    Phase 3: Refit the lens galaxy's bulge and disk _LightProfile_'s using fixed mass and source instances from phase 2, 
    where we:

        1) Do not use priors from phase 1 for the lens's _LightProfile_, assuming the source light could bias them.
        2) Use the same bulge and disk _PriorModel_'s created or phase 1, which use the same _Setup_.
    """

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
        mass=phase2.result.instance.galaxies.lens.mass,
        shear=phase2.result.instance.galaxies.lens.shear,
        hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__light_bulge_disk__mass_source_fixed",
        folders=folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                sersic=phase2.result.instance.galaxies.source.sersic,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase3 = phase3.extend_with_multiple_hyper_phases(setup_hyper=slam.setup_hyper)

    """
    Phase 4: Simultaneously fit the lens and source galaxies, where we:

        1) Set lens's light, mass, shear and source's light using models from phases 1 and 2.
    """

    phase4 = al.PhaseImaging(
        phase_name="phase_4__light_bulge_disk__mass_sie__source_sersic",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=phase3.result.instance.galaxies.lens.bulge,
                disk=phase3.result.instance.galaxies.lens.disk,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                sersic=phase2.result.model.galaxies.source.sersic,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase4 = phase4.extend_with_multiple_hyper_phases(setup_hyper=slam.setup_hyper)

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
