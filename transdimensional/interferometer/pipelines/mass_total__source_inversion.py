from os import path
import autofit as af
import autolens as al

"""
In this pipeline, we fit `Interferometer` of a strong lens system where:

 - The lens `Galaxy`'s light is omitted (and is not present in the simulated data).
 - The lens `Galaxy`'s total mass distribution is modeled as an input total `MassProfile` (default=`EllipticalPowerLaw`).
 - The source `Galaxy`'s surface-brightness is modeled using an `Inversion`.
.
The pipeline is four phases:

Phase 1:

    Fit the lens mass with an `EllipticalIsothermal` (and optional shear) and source bulge with an `EllipticalSersic`.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None.
    Notes: None.

Phase 2:

    Fit the source `Inversion` using the lens `EllipticalIsothermal` (and optional shear) inferred in phase 1. The
    `Pixelization` uses `SetupSourceInversion.pixelization_prior_model` (default=`Rectangular`) and 
    `Regulaization` uses `SetupSourceInversion.regularization_prior_model` (default=`Constant`).
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: SetupSourceInversion.pixelization_prior_model + SetupSourceInversion.regularization_prior_model
    Prior Passing: Lens Mass (instance -> phase1).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Phase 3:

    Refines the lens `EllipticalIsothermal` mass models using the source `Inversion` of phase 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: `Pixelization` (default=VoronoiMagnification) + `Regularization` (default=Constant)
    Prior Passing: Lens Mass (model -> phase 1), Source `Inversion` (instance -> phase 2)
    Notes: Lens mass varies, source `Inversion` parameters fixed.

Phase 4:

    Fit the `SetupMassTotal.mass_prior_model` (default=`EllipticalPowerLaw`) model, using priors from the  
    `EllipticalIsothermal` mass model of phase 3 and the source `Inversion` of phase 2.
    
    Lens Mass: SetupMassTotal.mass_prior_model + ExternalShear
    Source Light: SetupSourceInversion.pixelization_prior_model + SetupSourceInversion.regularization_prior_model
    Prior Passing: Lens Mass (model -> phase 3), Source `Inversion` (instance -> phase 3)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings, real_space_mask):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass[total]_source[inversion]"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an `ExternalShear`.
        2) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in phases 3 & 4).
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Phase 1: Fit the lens`s `MassProfile`'s and source `LightProfile`, where we:

        1) Use an `EllipticalIsothermal` for the lens's mass and `EllipticalSersic`for the source's bulge, 
           irrespective of the final model that is fitted by the pipeline.
        2) Include an `ExternalShear` in the mass model if `SetupMass.with_shear=True`.
    """

    phase1 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[1]_mass[sie]_source[bulge]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=al.mp.EllipticalIsothermal,
                shear=setup.setup_mass.shear_prior_model,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, bulge=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        real_space_mask=real_space_mask,
    )

    """
    Phase 2: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens`s `MassProfile`'s to the results of phase 1.
        2) Use the `Pixelization` input into `SetupSourceInversion.pixelization_prior_model`.
        3) Use the `Regularization` input into `SetupSourceInversion.regularization_prior_model`.
    """

    phase2 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[2]_mass[sie]_source[inversion_initialization]", n_live_points=20
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=setup.setup_source.pixelization_prior_model,
                regularization=setup.setup_source.regularization_prior_model,
            ),
        ),
        settings=settings,
        real_space_mask=real_space_mask,
    )

    """
    Phase 3: Refit the lens`s mass (and shear) using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Pixelization` and `Regularization` to the results of phase 2.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of phase 1.
    """

    phase3 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[3]_mass[sie]_source[inversion]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
        real_space_mask=real_space_mask,
    )

    """
    We also extend phase 3 with an additional `inversion_phase` which uses the best-fit mass model of phase 3 above 
    to refine the `Inversion` used, by fitting only the `Pixelization` & `Regularization` parameters. This is 
    equivalent to phase 2 above, but makes the code shorter and more easy to read.

    The the `inversion_phase` results are accessible as attributes of the phase results and used in phase 4 below.
    """

    phase3 = phase3.extend_with_inversion_phase(
        hyper_search=af.DynestyStatic(n_live_points=50)
    )

    """
    Phase 4: Fit the lens`s `MassProfile`'s with the input `SetupMassTotal.mass_prior_model` using the source 
    `Inversion` of phase above, where we:

        1) Use the source `Pixelization` and `Regularization inferred in phase 3`s extended `inversion_phase`.
        2) Set priors on the lens galaxy mass using the `EllipticalIsothermal` (and `ExternalShear`) of phase 3.
    """

    """
    If the `mass_prior_model` is an `EllipticalPowerLaw` `MassProfile` we can initialize its priors from the 
    `EllipticalIsothermal` fitted previously. If it is not an `EllipticalPowerLaw` we omit this setting up of
    priors, still benefitting from the initialized `Inversion` parameters..
    """

    mass = setup.setup_mass.mass_prior_model

    if mass.cls is al.mp.EllipticalPowerLaw:

        mass.centre = phase3.result.model.galaxies.lens.mass.centre
        mass.elliptical_comps = phase3.result.model.galaxies.lens.mass.elliptical_comps
        mass.einstein_radius = phase3.result.model.galaxies.lens.mass.einstein_radius

    phase4 = al.PhaseInterferometer(
        search=af.DynestyStatic(
            name="phase[4]_mass[total]_source[inversion]", n_live_points=100
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase3.result.inversion.instance.galaxies.source.pixelization,
                regularization=phase3.result.inversion.instance.galaxies.source.regularization,
            ),
        ),
        hyper_background_noise=af.last.hyper.instance.optional.hyper_background_noise,
        settings=settings,
        real_space_mask=real_space_mask,
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, phase1, phase2, phase3, phase4
    )
