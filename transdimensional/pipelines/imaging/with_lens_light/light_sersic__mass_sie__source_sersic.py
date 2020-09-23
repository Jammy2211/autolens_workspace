import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` of a strong lens system where:

 - The lens galaxy`s `LightProfile` is modeled as an _EllipticalSersic_.
 - The lens galaxy`s `MassProfile` is modeled as an _EllipticalIsothermal_.
 - The source galaxy`s `LightProfile` is modeled as an _EllipticalSersic_.

The pipeline is three phases:

Phase 1:

    Fit and subtract the lens light model.
    
    Lens Light: EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source _LightProfile_.
    
    Lens Light: EllipticalSersic
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the lens subtracted image from phase 1.

Phase 3:

    Refine the lens light and mass models and source light model using priors initialized from phases 1 and 2.
    
    Lens Light: EllipticalSersic
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> phase 1), lens mass and source light (model -> phase 2).
    Notes: None
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__light_sersic__mass_sie__source_sersic"

    """
    This pipeline is tagged according to whether:

    1) The lens galaxy mass model includes an  _ExternalShear_.
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """Setup: Include an `ExternalShear` in the mass model if turned on in _SetupMass_. """

    if not setup.setup_mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    """
    Phase 1: Fit only the lens galaxy`s light, where we:

        1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    """

    light = af.PriorModel(al.lp.EllipticalSersic)
    light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    light.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_sersic",
        folders=setup.folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=setup.redshift_lens, sersic=light)),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    Phase 2: Fit the lens`s `MassProfile``s and source galaxy`s light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
        2) Set priors on the centre of the lens galaxy`s `MassProfile` by linking them to those inferred for 
           the `LightProfile` in phase 1.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = phase1.result.model.galaxies.lens.light.centre_0
    mass.centre_1 = phase1.result.model.galaxies.lens.light.centre_1

    phase2 = al.PhaseImaging(
        phase_name="phase_2__mass_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                sersic=phase1.result.instance.galaxies.lens.sersic,
                mass=mass,
                shear=shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=60),
    )

    """
    Phase 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens`s light, mass, and source`s light using the results of phases 1 and 2.
    """

    phase3 = al.PhaseImaging(
        phase_name="phase_3__light_sersic__mass_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                sersic=phase1.result.model.galaxies.lens.sersic,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                sersic=phase2.result.model.galaxies.source.sersic,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
