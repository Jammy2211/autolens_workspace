import autofit as af
import autolens as al

"""
In this pipeline, we fit the a strong lens using a power-law mass proflie and a source which uses an
inversion.

The first 3 phases are identical to the pipeline 'lens_sie__source_inversion.py'.

The pipeline is four phases:

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

    Refines the lens light and SIE mass models using the source inversion of phase 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: pixelization (default=VoronoiMagnification) + regularization (default=Constant)
    Prior Passing: Lens Mass (model -> phase 1), Source Inversion (instance -> phase 2)
    Notes: Lens mass varies, source inversion parameters fixed.

Phase 4:

    Fit the power-law mass model, using priors from the SIE mass model of phase 3 and a source inversion.
    
    Lens Mass: EllipticalPowerLaw + ExternalShear
    Source Light: pixelization (default=VoronoiMagnification) + regularization (default=Constant)
    Prior Passing: Lens Mass (model -> phase 3), Source Inversion (instance -> phase 3)
    Notes: Lens mass varies, source inversion parameters fixed.
"""


def make_pipeline(
    setup, settings, redshift_lens=0.5, redshift_source=1.0, evidence_tolerance=5.0
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__lens_power_law__source_inversion"

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
            n_live_points=50, evidence_tolerance=evidence_tolerance
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
            n_live_points=20, evidence_tolerance=evidence_tolerance
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
        search=af.DynestyStatic(
            n_live_points=50, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Extend phase 3 with an additional 'inversion phase' which uses the best-fit mass model of phase 3 above
    to refine the it inversion used, by fitting only the pixelization & regularization parameters. This is equivalent
    to phase 2 above, but makes the code shorter and more easy to read.

    The the inversion phase results are accessible as attributes of the phase results and used in phase 4 below.
    """

    phase3 = phase3.extend_with_inversion_phase(
        inversion_search=af.DynestyStatic(n_live_points=101)
    )

    """
    Phase 4: Fit the lens galaxy's mass with a power-law and a source inversion, where we:

        1) Use the source inversion of phase 3's extended inversion phase.
        2) Set priors on the lens galaxy mass using the EllipticalIsothermal and ExternalShear of phase 3.
    """

    """Setup the power-law _MassProfile_ and initialize its priors from the SIE."""

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = phase3.result.model.galaxies.lens.mass.centre
    mass.elliptical_comps = phase3.result.model.galaxies.lens.mass.elliptical_comps
    mass.einstein_radius = phase3.result.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_power_law__source_inversion",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase3.result.inversion.instance.galaxies.source.pixelization,
                regularization=phase3.result.inversion.instance.galaxies.source.regularization,
            ),
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
