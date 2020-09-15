import autofit as af
import autolens as al

"""
In this pipeline, we fit _Interferometer_ data of a strong lens system where:

 - The lens galaxy's _LightProfile_ is omitted (and is not present in the simulated data).
 - The lens galaxy's _MassProfile_ is modeled as an _EllipticalIsothermal_.
 - The source galaxy's _LightProfile_ is modeled as 2 _EllipticalSersic_'s 

The pipeline is two phases:

Phase 1:

    Fit the lens mass model and source _LightProfile_ using x1 Sersic.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source _LightProfile_ using x1 source galaxies.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Galaxy 1 - Light: EllipticalSersic
    Source Galaxy 2 - Light: EllipticalSersic
    Prior Passing: Lens mass (model -> phase 1), Source Galaxy 1 Light (model -> phase 1)
    Notes: None
"""


def make_pipeline(setup, settings, real_space_mask):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__mass_sie__source_sersic_x2"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an  _ExternalShear_.
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """Setup: Include an _ExternalShear_ in the mass model if turned on in _SetupMass_. """

    if not setup.setup_mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    """
    Phase 1: Fit the lens's _MassProfile_'s and source _LightProfile_, where we:

        1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1__mass_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=setup.redshift_lens, mass=mass, shear=shear),
            source_0=al.GalaxyModel(
                redshift=setup.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        real_space_mask=real_space_mask,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    Phase 2: Fit the lens's _MassProfile_'s and two source galaxies, where we:

        1) Set the priors on the lens galaxy _MassProfile_'s using the results of phase 1.
        2) Set the priors on the first source galaxy's light using the results of phase 1.
    """

    phase2 = al.PhaseInterferometer(
        phase_name="phase_2__mass_sie__source_sersic_x2",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source_0=al.GalaxyModel(
                redshift=setup.redshift_source,
                sersic=phase1.result.model.galaxies.source_0.sersic,
            ),
            source_1=al.GalaxyModel(
                redshift=setup.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        real_space_mask=real_space_mask,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
