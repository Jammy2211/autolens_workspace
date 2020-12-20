from os import path
import autofit as af
import autolens as al

"""
This pipeline performs a parametric source analysis which fits a lens model (the lens`s `LightProfile` and mass) and the
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

    Fit the lens mass model and source `LightProfile`, using the lens subtracted image from phase 1.
    
    Lens Light: None
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: None
    Notes: Uses the lens light subtracted image from phase 1

Phase 3:

    Refit the lens `LightProfile` using the mass model and source `LightProfile` fixed from phase 2.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: lens mass and source (instance -> phase 2)
    Notes: None

Phase 4:

    Refine the lens `LightProfile` and `MassProfile` and source `LightProfile`, using priors from the previous 2 phases.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: Lens light (model -> phase 3), lens mass and source (model -> phase 2)
    Notes: None
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source[parametric]"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """

    path_prefix = path.join(slam.path_prefix, pipeline_name, slam.source_parametric_tag)

    """
    Phase 1: Fit only the lens `Galaxy`'s light, where we:

        1) Use the light model determined from `SetupLightParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.).
    """
    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=slam.pipeline_source_parametric.setup_light.bulge_prior_model,
        disk=slam.pipeline_source_parametric.setup_light.disk_prior_model,
        envelope=slam.pipeline_source_parametric.setup_light.envelope_prior_model,
    )

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[1]_light[parametric]", n_live_points=75),
        galaxies=af.CollectionPriorModel(lens=lens),
        settings=settings,
    )

    phase1 = phase1.extend_with_hyper_phase(
        setup_hyper=slam.setup_hyper, include_hyper_image_sky=False
    )

    """
    Phase 2: Fit the lens`s `MassProfile`'s and source `Galaxy`'s `LightProfile`, where we:

        1) Fix the foreground lens `LightProfile` to the result of phase 1.
        2) Set priors on the centre of the lens `Galaxy`'s total mass distribution by linking them to those inferred for
           the bulge of the `LightProfile` in phase 1.
        3) The source model determined from `SetupSourceParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.)
    """

    mass = slam.pipeline_source_parametric.setup_mass.mass_prior_model

    """SLaM: Align the bulge and mass model centres if align_bulge_mass_centre is True."""

    if slam.pipeline_source_parametric.setup_mass.mass_centre is None:
        if slam.pipeline_source_parametric.setup_mass.align_bulge_mass_centre:
            mass.centre = phase1.result.instance.galaxies.lens.bulge.centre
        else:
            mass.centre = phase1.result.model.galaxies.lens.bulge.centre

    """SLaM: The shear model is chosen below based on the input of `SetupSourceParametric`."""

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_light[fixed]_mass[total]_source[parametric]",
            n_live_points=200,
            walks=10,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=phase1.result.instance.galaxies.lens.bulge,
                disk=phase1.result.instance.galaxies.lens.disk,
                envelope=phase1.result.instance.galaxies.lens.envelope,
                mass=mass,
                shear=slam.pipeline_source_parametric.setup_mass.shear_prior_model,
                hyper_galaxy=phase1.result.hyper.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                bulge=slam.pipeline_source_parametric.setup_source.bulge_prior_model,
                disk=slam.pipeline_source_parametric.setup_source.disk_prior_model,
                envelope=slam.pipeline_source_parametric.setup_source.envelope_prior_model,
                hyper_galaxy=phase1.result.hyper.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_background_noise=phase1.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    phase2 = phase2.extend_with_hyper_phase(
        setup_hyper=slam.setup_hyper, include_hyper_image_sky=False
    )

    """
    Phase 3: Refit the lens `Galaxy`'s bulge and disk `LightProfile`'s using fixed mass and source instances from phase 2, 
    where we:

        1) Use the light model determined from `SetupLightParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.).
        2) Do not use priors from phase 1 for the lens`s `LightProfile`, assuming the source light could bias them.
    """

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=slam.pipeline_source_parametric.setup_light.bulge_prior_model,
        disk=slam.pipeline_source_parametric.setup_light.disk_prior_model,
        envelope=slam.pipeline_source_parametric.setup_light.envelope_prior_model,
        mass=phase2.result.instance.galaxies.lens.mass,
        shear=phase2.result.instance.galaxies.lens.shear,
        hyper_galaxy=phase2.result.hyper.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_light[parametric]_mass[fixed]_source[fixed]",
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=lens,
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                bulge=phase2.result.instance.galaxies.source.bulge,
                disk=phase2.result.instance.galaxies.source.disk,
                envelope=phase2.result.instance.galaxies.source.envelope,
                hyper_galaxy=phase2.result.hyper.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_background_noise=phase2.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    phase3 = phase3.extend_with_hyper_phase(
        setup_hyper=slam.setup_hyper, include_hyper_image_sky=False
    )

    """
    Phase 4: Simultaneously fit the lens and source galaxies, where we:

        1) Set lens`s light, mass, shear and source`s light using models from phases 2 and 3.
    """

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_light[parametric]_mass[total]_source[parametric]",
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=phase3.result.model.galaxies.lens.bulge,
                disk=phase3.result.model.galaxies.lens.disk,
                envelope=phase3.result.model.galaxies.lens.envelope,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase3.result.hyper.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                bulge=phase2.result.model.galaxies.source.bulge,
                disk=phase2.result.model.galaxies.source.disk,
                envelope=phase2.result.model.galaxies.source.envelope,
                hyper_galaxy=phase3.result.hyper.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_background_noise=phase3.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    phase4 = phase4.extend_with_hyper_phase(
        setup_hyper=slam.setup_hyper, include_hyper_image_sky=True
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, None, phase1, phase2, phase3, phase4
    )
