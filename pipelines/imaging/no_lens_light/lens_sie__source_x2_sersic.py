import autofit as af
import autolens as al

"""
In this pipeline, we fit the a strong lens using a _EllipticalIsothermal_ mass profile and a source which uses two Sersic profiles.

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


def make_pipeline(
    setup, settings, redshift_lens=0.5, redshift_source=1.0, evidence_tolerance=0.8
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__lens_sie__source_x2_sersic"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    # This pipeline is tagged according to whether:

    # 1) The lens galaxy mass model includes an external shear.

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
        phase_name="phase_1__lens_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, mass=mass, shear=shear),
            source_0=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=50, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 2: Fit the lens galaxy's mass and two source galaxies, where we:

        1) Set the priors on the lens galaxy mass using the results of phase 1.
        2) Set the priors on the first source galaxy's light using the results of phase 1.
    """

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_x2_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source_0=al.GalaxyModel(
                redshift=redshift_source,
                light=phase1.result.model.galaxies.source_0.light,
            ),
            source_1=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
