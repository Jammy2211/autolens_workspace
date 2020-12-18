from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` of a strong lens system where:

 - The lens `Galaxy`'s light is modeled parametrically using one or more input `LightProfile`s.
 - The lens `Galaxy`'s light matter mass distribution is fitted with the `LightProfile`'s of the 
      lens's light, where they are converted to a stellar mass distribution via constant mass-to-light ratios.
 - The lens `Galaxy`'s dark matter mass distribution is modeled as a _SphericalNFW_.
 - The source `Galaxy`'s surface-brightness is modeled using an `Inversion`.

The pipeline is five phases:

Phase 1:

    Fit and subtract the lens light with the parametric profiles input into `SetupLightParametric` (e.g. the 
    `bulge_prior_model`, `disk_prior_model`, etc). The default is :
    
    - `SetupLightParametric.bulge_prior_model=EllipticalSersic`, 
    - `SetupLightParametric.disk_prior_model=EllipticalExponential`
    - `SetupLightParametric.align_bulge_disk_centre=True` (meaning the two profiles above have aligned centre.
    
    Lens Light: SetupLightParametric.bulge_prior_model + SetupLightParametric.disk_prior_model + others
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source `LightProfile`, where the `LightProfile` parameters of the lens`s 
    `LightMassProfile` are fixed to the results of phase 1.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the lens subtracted image from phase 1.

Phase 3:

    Refine the lens light and mass models and source light model using priors initialized from phases 1 and 2.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> phase 1), lens mass and source light (model -> phase 2).
    Notes: None

Phase 4:

    Fit the source `Inversion` using the lens light and `MassProfile`'s inferred in phase 3.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (instance -> phase3).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Phase 5:

    Refines the lens light and mass models using the source `Inversion` of phase 4.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (model -> phase 3), Source `Inversion` (instance -> phase 4)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_light[parametric]_mass[light_dark]_source[inversion]"

    """
    This pipeline is tagged according to whether:

        1) The bulge model, disk model, etc. and whether their centres / elliptical_comps are aligned.
        2) If the bulge and disk share the same mass-to-light ratio or each is fitted independently.
        3) The lens galaxy mass model includes an `ExternalShear`.
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Phase 1: Fit only the lens `Galaxy`'s light, where we:

        1) Use the light model determined from `SetupLightParametric` (e.g. `bulge_prior_model`, `disk_prior_model`, 
           etc.)
    """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(name="phase[1]_light[parametric]", n_live_points=50),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=setup.setup_light.bulge_prior_model,
                disk=setup.setup_light.disk_prior_model,
                envelope=setup.setup_light.envelope_prior_model,
            )
        ),
        settings=settings,
    )

    """
    Phase 2: Fit the lens`s `MassProfile`'s and source `Galaxy`'s light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy bulge+disk model from phase 1.
        2) Set priors on the centre of the lens `Galaxy`'s dark matter mass distribution by linking them to those inferred 
           for the `LightProfile` in phase 1.
        3) Use a `SphericalNFWMCRLudlow` model for the dark matter which sets its scale radius via a mass-concentration
           relation and the lens and source redshifts.
    """

    bulge = setup.setup_mass.bulge_prior_instance_with_updated_priors()
    disk = setup.setup_mass.disk_prior_instance_with_updated_priors()
    envelope = setup.setup_mass.envelope_prior_instance_with_updated_priors()

    dark = setup.setup_mass.dark_prior_model
    dark.mass_at_200 = af.LogUniformPrior(lower_limit=5e8, upper_limit=5e14)
    dark.redshift_object = setup.redshift_lens
    dark.redshift_source = setup.redshift_source

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_light[fixed]_mass[light_dark]_source[bulge]",
            n_live_points=60,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=bulge,
                disk=disk,
                envelope=envelope,
                dark=dark,
                shear=setup.setup_mass.shear_prior_model,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, bulge=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
    )

    """
    Phase 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens`s bulge, disk, dark, and source`s light using the results of phases 1 and 2.
    """

    bulge = setup.setup_mass.light_and_mass_prior_models_with_updated_priors()
    disk = setup.setup_mass.disk_prior_model_with_updated_priors()
    envelope = setup.setup_mass.envelope_prior_model_with_updated_priors()

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_light[parametric]_mass[light_dark]_source[bulge]",
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=bulge,
                disk=disk,
                envelope=envelope,
                dark=phase2.result.model.galaxies.lens.dark,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                bulge=phase2.result.model.galaxies.source.bulge,
            ),
        ),
        settings=settings,
    )

    """
    Phase 4: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens`s bulge, disk and mass model to the results of phase 3.
    """

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_light[fixed]_mass[fixed]_source[inversion_initialization]",
            n_live_points=20,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=phase3.result.instance.galaxies.lens.bulge,
                disk=phase3.result.instance.galaxies.lens.disk,
                envelope=phase3.result.instance.galaxies.lens.envelope,
                dark=phase3.result.instance.galaxies.lens.dark,
                shear=phase3.result.instance.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=setup.setup_source.pixelization_prior_model,
                regularization=setup.setup_source.regularization_prior_model,
            ),
        ),
        settings=settings,
    )

    """
    Phase 5: Fit the lens`s bulge, disk and mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of phase 4.
        2) Set priors on the lens galaxy bulge, disk and mass using the results of phase 3.
    """

    phase5 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[5]_light[parametric]_mass[light_dark]_source[inversion]",
            n_live_points=100,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=phase3.result.model.galaxies.lens.bulge,
                disk=phase3.result.model.galaxies.lens.disk,
                envelope=phase3.result.model.galaxies.lens.envelope,
                dark=phase3.result.model.galaxies.lens.dark,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, phase1, phase2, phase3, phase4, phase5
    )
