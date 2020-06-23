import autofit as af
import autolens as al

"""
In this pipeline, we'll perform a parametric source analysis which fits a lens model (the lens's light, mass and
source's light). The lens's light is modeled as a sum of elliptical Gaussian profiles. This pipeline uses four phases:

Phase 1:

Fit and subtract the lens light model.

Lens Light: EllipticalGaussian(s)
Lens Mass: None
Source Light: None
Previous Pipelines: None
Prior Passing: None
Notes: None

Phase 2:

Fit the lens mass model and source _LightProfile_, using the lens subtracted image from phase 1.

Lens Light: None
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: EllipticalSersic
Previous Pipelines: None
Prior Passing: None
Notes: Uses the lens light subtracted image from phase 1

Phase 3:

Refit the lens light models using the mass model and source _LightProfile_ fixed from phase 2.

Lens Light: EllticalGaussian(s)
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: EllipticalSersic
Previous Pipelines: None
Prior Passing: lens mass and source (instance -> phase 2)
Notes: None

Phase 4:

Refine the lens light and mass models and source _LightProfile_, using priors from the previous 2 phases.

Lens Light: EllipticalGaussian(s)
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: EllipticalSersic
Previous Pipelines: None
Prior Passing: Lens light (model -> phase 3), lens mass and source (model -> phase 2)
Notes: None
"""


def make_pipeline(
    setup, settings, redshift_lens=0.5, redshift_source=1.0, evidence_tolerance=100.0
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__parametric__lens_gaussians"

    """
    This pipeline is tagged according to whether:

    1) Hyper-fitting settings (galaxies, sky, background noise) are used.
    2) The lens galaxy mass model includes an external shear.
    """
    # 3) The number of Gaussians in the lens light model.

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """
    Phase 1: Fit only the lens galaxy's light, where we:

    1) Align the Gaussian's (y,x) centres.
    """

    gaussian_0 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_1 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_2 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_3 = af.PriorModel(al.lp.EllipticalGaussian)

    """Align the centres of all gaussians."""

    gaussian_1.centre = gaussian_0.centre
    gaussian_2.centre = gaussian_0.centre
    gaussian_3.centre = gaussian_0.centre

    """Align the light model centres (all Gaussians) with the input setup lens_light_centre, if input."""

    if setup.lens_light_centre is not None:
        gaussian_0.centre = setup.lens_light_centre
        gaussian_1.centre = setup.lens_light_centre
        gaussian_2.centre = setup.lens_light_centre
        gaussian_3.centre = setup.lens_light_centre

    gaussian_0.add_assertion(gaussian_0.sigma < gaussian_1.sigma)
    gaussian_0.add_assertion(gaussian_1.sigma < gaussian_2.sigma)
    gaussian_0.add_assertion(gaussian_2.sigma < gaussian_3.sigma)

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        gaussian_0=gaussian_0,
        gaussian_1=gaussian_1,
        gaussian_2=gaussian_2,
        gaussian_3=gaussian_3,
    )

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_gaussians",
        folders=setup.folders,
        galaxies=dict(lens=lens),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=40, facc=0.3, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 2: Fit the lens galaxy's mass and source galaxy's light, where we:

    1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
    2) Set priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for \
       the bulge of the _LightProfile_ in phase 1.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    """Align the mass model centre with the input slam value, if input."""

    if setup.lens_mass_centre is not None:
        mass.centre = setup.lens_mass_centre

    """SETUP SHEAR: Include the shear in the mass model if not switched off in the pipeline setup. """

    if not setup.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                gaussian_0=phase1.result.instance.galaxies.lens.gaussian_0,
                gaussian_1=phase1.result.instance.galaxies.lens.gaussian_1,
                gaussian_2=phase1.result.instance.galaxies.lens.gaussian_2,
                gaussian_3=phase1.result.instance.galaxies.lens.gaussian_3,
                mass=mass,
                shear=shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=50, facc=0.5, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 3: Refit the lens galaxy's light using a fixed mass and source model above, where we:

    1) Do not use priors from phase 1 to Fit the lens's light, assuming the source light may of impacted them.
    """

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        gaussian_0=gaussian_0,
        gaussian_1=gaussian_1,
        gaussian_2=gaussian_2,
        gaussian_3=gaussian_3,
        mass=phase2.result.instance.galaxies.lens.mass,
        shear=phase2.result.instance.galaxies.lens.shear,
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_gaussians_sie__source_fixed",
        folders=setup.folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                light=phase2.result.instance.galaxies.source.light,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=75, facc=0.5, evidence_tolerance=evidence_tolerance
        ),
    )

    """
    Phase 4: Fit simultaneously the lens and source galaxies, where we:

    1) Set lens's light, mass, shear and source's light using the results of phases 1 and 2.
    """

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_gaussians_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                gaussian_0=phase3.result.model.galaxies.lens.gaussian_0,
                gaussian_1=phase3.result.model.galaxies.lens.gaussian_1,
                gaussian_2=phase3.result.model.galaxies.lens.gaussian_2,
                gaussian_3=phase3.result.model.galaxies.lens.gaussian_3,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                light=phase2.result.model.galaxies.source.light,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=75, facc=0.3),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
