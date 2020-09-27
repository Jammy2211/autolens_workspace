import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` of a strong lens system where:

 - The lens `Galaxy`'s `LightProfile` is omitted (and is not present in the simulated data).
 - The lens `Galaxy`'s `MassProfile` is modeled as an _EllipticalPowerLaw_.
 - The source `Galaxy`'s surface-brightness is modeled using an _Inversion_.

The first 3 phases are identical to the pipeline `lens_sie__source_inversion.py`.

The pipeline is four phases:

Phase 1:

    Fit the lens mass model and source _LightProfile_.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None.
    Notes: None.

Phase 2:

    Fit the source `Inversion` using the lens `MassProfile` inferred in phase 1.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens & Mass (instance -> phase1).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Phase 3:

    Refines the lens light and  `EllipticalIsothermal` mass models using the source `Inversion` of phase 2.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: `Pixelization` (default=VoronoiMagnification) + `Regularization` (default=Constant)
    Prior Passing: Lens Mass (model -> phase 1), Source `Inversion` (instance -> phase 2)
    Notes: Lens mass varies, source `Inversion` parameters fixed.

Phase 4:

    Fit the `EllipticalPowerLaw` mass model, using priors from the  `EllipticalIsothermal` mass model of phase 3 and 
    a source _Inversion_.
    
    Lens Mass: EllipticalPowerLaw + ExternalShear
    Source Light: `Pixelization` (default=VoronoiMagnification) + `Regularization` (default=Constant)
    Prior Passing: Lens Mass (model -> phase 3), Source `Inversion` (instance -> phase 3)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__mass_power_law__source_inversion"

    """
    This pipeline is tagged according to whether:

        1) The lens galaxy mass model includes an  _ExternalShear_.
        2) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in phases 3 & 4).
    """

    path_prefix = f"{setup.path_prefix}/{pipeline_name}/{setup.tag}"

    """Setup: Include an `ExternalShear` in the mass model if turned on in _SetupMass_. """

    if not setup.setup_mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    """
    Phase 1: Fit the lens`s `MassProfile`'s and source `LightProfile`, where we:

        1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        path_prefix=path_prefix,
        phase_name="phase_1__source_sersic",
        galaxies=dict(
            lens=al.GalaxyModel(redshift=setup.redshift_lens, mass=mass, shear=shear),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    Phase 2: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens`s `MassProfile`'s to the results of phase 1.
    """

    phase2 = al.PhaseImaging(
        path_prefix=path_prefix,
        phase_name="phase_2__source_inversion_initialization",
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
        settings=settings,
        search=af.DynestyStatic(n_live_points=20),
    )

    """
    Phase 3: Fit the lens`s mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of phase 2.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of phase 1.
    """

    phase3 = al.PhaseImaging(
        path_prefix=path_prefix,
        phase_name="phase_3__source_inversion",
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
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    Extend phase 3 with an additional `inversion phase` which uses the best-fit mass model of phase 3 above to refine 
    the `Inversion` used, by fitting only the `Pixelization` & `Regularization` parameters. This is equivalent
    to phase 2 above, but makes the code shorter and more easy to read.

    The the `Inversion` phase results are accessible as attributes of the phase results and used in phase 4 below.
    """

    phase3 = phase3.extend_with_inversion_phase(
        inversion_search=af.DynestyStatic(n_live_points=50)
    )

    """
    Phase 4: Fit the lens`s `MassProfile`'s with a `EllipticalPowerLaw` and a source `Inversion`, where we:

        1) Use the source `Inversion` of phase 3`s extended `Inversion` phase.
        2) Set priors on the lens galaxy `MassProfile`'s using the `EllipticalIsothermal` and `ExternalShear` of phase 3.
    """

    """
    Setup the `EllipticalPowerLaw` `MassProfile` and initialize its priors from the _EllipticalIsothermal_.
    
    Note: Because the `EllipticalIsothermal` `MassProfile` does not have a slope, to pass its prior to the 
    `EllipticalPowerLaw` we set it up as a `PriorModel` and pass its parameters one-by-one.
    """

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = phase3.result.model.galaxies.lens.mass.centre
    mass.elliptical_comps = phase3.result.model.galaxies.lens.mass.elliptical_comps
    mass.einstein_radius = phase3.result.model.galaxies.lens.mass.einstein_radius

    phase4 = al.PhaseImaging(
        path_prefix=path_prefix,
        phase_name="phase_4__mass_power_law__source_inversion",
        galaxies=dict(
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
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
