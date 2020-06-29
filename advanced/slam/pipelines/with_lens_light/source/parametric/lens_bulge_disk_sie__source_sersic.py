import autofit as af
import autolens as al

"""
In this pipeline, we'll perform a parametric source analysis which fits a lens model (the lens's light, mass and
source's light). This pipeline uses four phases:

Phase 1:

    Fit and subtract the lens light model.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: None
    Source Light: None
    Previous Pipelines: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source _LightProfile_, using the lens subtracted image from phase 1.
    
    Lens Light: None
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: None
    Notes: Uses the lens light subtracted image from phase 1

Phase 3:

    Refit the lens light models using the mass model and source _LightProfile_ fixed from phase 2.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: lens mass and source (instance -> phase 2)
    Notes: None

Phase 4:

    Refine the lens light and mass models and source _LightProfile_, using priors from the previous 2 phases.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: Lens light (model -> phase 3), lens mass and source (model -> phase 2)
    Notes: None
"""


def make_pipeline(slam, settings, redshift_lens=0.5, redshift_source=1.0):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__parametric"

    """For pipeline tagging we set the source and lens light types."""
    slam.set_source_type(source_type="sersic")
    slam.set_light_type(light_type="")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an external shear.
    """

    folders = slam.folders + [pipeline_name, slam.source_pipeline_tag, slam.source.tag]

    """
    Phase 1: Fit only the lens galaxy's light, where we:

        1) Align the bulge and disk (y,x) centre.
    """

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lp.EllipticalSersic,
        disk=al.lp.EllipticalExponential,
    )

    lens.bulge.centre = lens.disk.centre

    """SLaM: Align the light model centres (bulge and disk) with the input slam lens_light_centre, if input."""

    lens.bulge = slam.source.align_centre_to_lens_light_centre(light=lens.bulge)
    lens.disk = slam.source.align_centre_to_lens_light_centre(light=lens.disk)

    """SLaM: Remove the disk from the lens light model if lens_light_bulge_only is True."""

    lens = slam.source.remove_disk_from_lens_galaxy(lens=lens)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_bulge_disk",
        folders=folders,
        galaxies=dict(lens=lens),
        settings=settings,
        search=af.DynestyStatic(n_live_points=75),
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(setup=slam.hyper)

    """
    Phase 2: Fit the lens galaxy's mass and source galaxy's light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
        2) Set priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for
           the bulge of the _LightProfile_ in phase 1.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    """SLaM: Align the light and mass model centres if align_light_mass_centre is True."""

    if slam.source.align_light_mass_centre:
        mass.centre = phase1.result.instance.galaxies.lens.bulge.centre
    else:
        mass.centre = phase1.result.model.galaxies.lens.bulge.centre

    """SLaM: Align the mass model centre with the input slam lens_mass_centre, if input."""

    mass = slam.source.align_centre_to_lens_mass_centre(mass=mass)

    """SLaM: The shear model is chosen below based on the settings of the slam source."""

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase1.result.instance.galaxies.lens.bulge,
                disk=phase1.result.instance.galaxies.lens.disk,
                mass=mass,
                shear=slam.source.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=al.lp.EllipticalSersic,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase2 = phase2.extend_with_multiple_hyper_phases(setup=slam.hyper)

    """
    Phase 3: Refit the lens galaxy's light using fixed mass and source instances from phase 2, where we:

        1) Do not use priors from phase 1 to Fit the lens's light, assuming the source light may of impacted them.
    """

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lp.EllipticalSersic,
        disk=al.lp.EllipticalExponential,
        mass=phase2.result.instance.galaxies.lens.mass,
        shear=phase2.result.instance.galaxies.lens.shear,
        hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    lens.bulge.centre = lens.disk.centre

    slam.source.align_centre_to_lens_light_centre(light=lens.bulge)
    slam.source.align_centre_to_lens_light_centre(light=lens.bulge)

    if slam.source.lens_light_bulge_only:
        lens.disk = None

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_bulge_disk_sie__source_fixed",
        folders=folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase2.result.instance.galaxies.source.sersic,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase3 = phase3.extend_with_multiple_hyper_phases(setup=slam.hyper)

    """
    Phase 4: Simultaneously fit the lens and source galaxies, where we:

        1) Set lens's light, mass, shear and source's light using models from phases 1 and 2.
    """

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_bulge_disk_sie__source_sersic",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase3.result.model.galaxies.lens.bulge,
                disk=phase3.result.model.galaxies.lens.disk,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase2.result.model.galaxies.source.sersic,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase4 = phase4.extend_with_multiple_hyper_phases(setup=slam.hyper)

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
