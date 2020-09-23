import autofit as af
import autolens as al

"""
In this pipeline, we fit `Interferometer` data of a strong lens system where:

 - The lens galaxy`s `LightProfile` is omitted (and is not present in the simulated data).
 - The lens galaxy`s `MassProfile` is modeled as an _EllipticalIsothermal_.
 - The source galaxy`s surface-brightness is modeled using an _Inversion_.

The pipeline is three phases:

Phase 1:

    Fit the lens `MassProfile``s and source _LightProfile_.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None.
    Notes: None.

Phase 2:

    Fit the source `Inversion` using the lens `MassProfile``s inferred in phase 1.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens & Mass (instance -> phase1).
    Notes: Lens `MassProfile``s fixed, source `Inversion` parameters vary.

Phase 3:

    Refines the lens `MassProfile``s using the source `Inversion` of phase 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Mass (model -> phase 1), Source `Inversion` (instance -> phase 2)
    Notes: Lens `MassProfile``s varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings, real_space_mask):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__mass_sie__source_inversion"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an  _ExternalShear_.
        2) The `Pixelization` and `Regularization` scheme of the pipeline (use in phase 3).
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """Setup: Include an `ExternalShear` in the mass model if turned on in _SetupMass_. """

    if not setup.setup_mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    """
    Phase 1: Fit the lens`s `MassProfile``s and source `LightProfile`, where we:

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
            source=al.GalaxyModel(
                redshift=setup.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        real_space_mask=real_space_mask,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    Phase 2: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens`s `MassProfile``s to the results of phase 1.
    """

    phase2 = al.PhaseInterferometer(
        phase_name="phase_2__source_inversion_initialization",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=setup.setup_source.pixelization,
                regularization=setup.setup_source.regularization,
            ),
        ),
        real_space_mask=real_space_mask,
        settings=settings,
        search=af.DynestyStatic(n_live_points=20),
    )

    """
    Phase 3: Fit the lens`s mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of phase 2.
        2) Set priors on the lens galaxy `MassProfile``s using the results of phase 1.
    """

    phase3 = al.PhaseInterferometer(
        phase_name="phase_3__mass_sie__source_inversion",
        folders=setup.folders,
        galaxies=dict(
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
        real_space_mask=real_space_mask,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
