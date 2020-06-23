import autofit as af
import autolens as al

"""
In this pipeline, we fit the a strong lens using a _EllipticalSersic_ _LightProfile_, _EllipticalIsothermal_ mass profile and a source which uses an
inversion.

The first 3 phases are identical to the pipeline 'lens_sersic_sie__source_sersic.py'.

The pipeline is five phases:

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

Phase 4:

Fit the source inversion using the lens light and _MassProfile_s inferred in phase 3.

Lens Light: EllipticalSersic
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: VoronoiMagnification + Constant
Prior Passing: Lens Light & Mass (instance -> phase3).
Notes: Lens mass fixed, source inversion parameters vary.

Phase 5:

Refines the lens light and mass models using the source inversion of phase 4.

Lens Light: EllipticalSersic
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: VoronoiMagnification + Constant
Prior Passing: Lens Light & Mass (model -> phase 3), Source Inversion (instance -> phase 4)
Notes: Lens mass varies, source inversion parameters fixed.
"""


def make_pipeline(
    setup, settings, redshift_lens=0.5, redshift_source=1.0, evidence_tolerance=100.0
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__lens_sersic_sie__source_inversion"

    """
    This pipeline is tagged according to whether:

    1) The lens galaxy mass model includes an external shear.
    2) The pixelization and regularization scheme of the pipeline (fitted in phases 4 & 5).
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """SETUP SHEAR: Include the shear in the mass model if not switched off in the pipeline setup. """

    if not setup.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    """
    Phase 1; Fit only the lens galaxy's light, where we:

    1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    """

    light = af.PriorModel(al.lp.EllipticalSersic)
    light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    light.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        folders=setup.folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=redshift_lens, light=light)),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=50, facc=0.5, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 2: Fit the lens galaxy's mass and source galaxy's light, where we:

    1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
    2) Set priors on the centre of the lens galaxy's _MassProfile_ by linking them to those inferred for \
       the _LightProfile_ in phase 1.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = phase1.result.model.galaxies.lens.light.centre_0
    mass.centre_1 = phase1.result.model.galaxies.lens.light.centre_1

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=phase1.result.instance.galaxies.lens.light,
                mass=mass,
                shear=shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=60, facc=0.2, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 3: Fit simultaneously the lens and source galaxies, where we:

    1) Set the lens's light, mass, and source's light using the results of phases 1 and 2.
    """

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=phase1.result.model.galaxies.lens.light,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                light=phase2.result.model.galaxies.source.light,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=75, facc=0.3, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 4: Fit the input pipeline pixelization & regularization, where we:

    1) Set lens's light and mass model using the results of phase 3.
    """

    phase4 = al.PhaseImaging(
        phase_name="phase_4__source_inversion_initialization",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=setup.pixelization,
                regularization=setup.regularization,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=20, facc=0.8, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 5: Fit the lens's mass using the input pipeline pixelization & regularization, where we:

    1) Fix the source inversion parameters to the results of phase 4.
    2) Set priors on the lens galaxy light and mass using the results of phase 3.
    """

    phase5 = al.PhaseImaging(
        phase_name="phase_5__lens_sersic_sie__source_inversion",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=phase3.result.model.galaxies.lens.light,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=60, facc=0.4),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4, phase5)
