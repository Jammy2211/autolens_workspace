import autofit as af
import autolens as al

"""
In this pipeline, we fit the a strong lens using a _EllipticalIsothermal_ mass profile and a source which uses an
inversion.

The pipeline is three phases:

Phase 1:

    Fit the lens mass model and source _LightProfile_.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None.
    Notes: None.

Phase 2:

    Fit the source inversion using the lens _MassProfile_ inferred in phase 1.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens & Mass (instance -> phase1).
    Notes: Lens mass fixed, source inversion parameters vary.

Phase 3:

    Refines the lens light and mass models using the source inversion of phase 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Mass (model -> phase 1), Source Inversion (instance -> phase 2)
    Notes: Lens mass varies, source inversion parameters fixed.
"""


def make_pipeline(
    setup, settings, redshift_lens=0.5, redshift_source=1.0, evidence_tolerance=100.0
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__lens_sie__source_inversion"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an external shear.
        2) The pixelization and regularization scheme of the pipeline (fitted in phases 3 & 4).
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """SETUP SHEAR: Include the shear in the mass model if not switched off in the pipeline setup. """

    if not setup.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    """
    Phase 1: Fit the lens galaxy's mass and source light, where we:

        1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, mass=mass, shear=shear),
            source=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=50,
            sampling_efficiency=0.5,
            evidence_tolerance=evidence_tolerance,
        ),
    )

    """
    Phase 2: Fit the input pipeline pixelization & regularization, where we:

        1) Set lens's mass model using the results of phase 1.
    """

    phase2 = al.PhaseImaging(
        phase_name="phase_2__source_inversion_initialization",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=setup.pixelization,
                regularization=setup.regularization,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=20,
            sampling_efficiency=0.8,
            evidence_tolerance=evidence_tolerance,
        ),
    )

    """
    Phase 3: Fit the lens's mass using the input pipeline pixelization & regularization, where we:

        1) Fix the source inversion parameters to the results of phase 2.
        2) Set priors on the lens galaxy mass using the results of phase 1.
    """

    phase3 = al.PhaseImaging(
        phase_name="phase_3__source_inversion",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50, sampling_efficiency=0.5),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
