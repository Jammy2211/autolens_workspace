import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` of a strong lens system where:

 - The lens galaxy`s `LightProfile` is modeled as an `EllipticalSersic` and _EllipticalExponential_.
 - The lens galaxy`s `MassProfile` is modeled as an `EllipticalIsothermal` and _ExternalShear_.
 - The source galaxy`s surface-brightness is modeled using an _Inversion_.

The pipeline is five phases:

Phase 1:

    Fit and subtract the lens light model.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source _LightProfile_.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the lens subtracted image from phase 1.

Phase 3:

    Refine the lens light and mass models and source light model using priors initialized from phases 1 and 2.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> phase 1), lens mass and source light (model -> phase 2).
    Notes: None

Phase 4:

    Fit the source `Inversion` using the lens light and `MassProfile``s inferred in phase 3.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (instance -> phase3).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Phase 5:

    Refines the lens light and mass models using the source `Inversion` of phase 4.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (model -> phase 3), Source `Inversion` (instance -> phase 4)
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__light_bulge_disk__mass_sie__source_inversion"

    """
    This pipeline is tagged according to whether:

        1) The bulge + disk centres or elliptical_comps are aligned.
        2) The disk component of the lens light model is an `EllipticalExponential` or `EllipticalSersic` profile.
        3) The lens galaxy mass model includes an  _ExternalShear_.
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """
    Phase 1; Fit only the lens galaxy`s light, where we:

        1) Set priors on the bulge and disk`s (y,x) centres such that we assume the image is centred around 
           the lens galaxy OR,
        2) Fix the lens light centre to the input value in _SetupLightBulgeDisk_.
    """

    bulge = af.PriorModel(al.lp.EllipticalSersic)

    """Setup: Set whether the disk is modeled as an `EllipticalSersic` or _EllipticalExponential_."""

    if setup.setup_light.disk_as_sersic:
        disk = af.PriorModel(al.lp.EllipticalSersic)
    else:
        disk = af.PriorModel(al.lp.EllipticalExponential)

    """Setup: Set the alignment of the bulge and disk`s centres and elliptical components."""

    if setup.setup_light.align_bulge_disk_centre:
        bulge.centre = disk.centre

    if setup.setup_light.align_bulge_disk_elliptical_comps:
        bulge.elliptical_comps = disk.elliptical_comps

    """Setup: Fix the bulge and disk centres to the input value in `SetupLight` if input."""

    if setup.setup_light.light_centre is not None:
        bulge.centre_0 = setup.setup_light.light_centre[0]
        bulge.centre_1 = setup.setup_light.light_centre[1]
        disk.centre_0 = setup.setup_light.light_centre[0]
        disk.centre_1 = setup.setup_light.light_centre[1]

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_bulge_disk",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=setup.redshift_lens, bulge=bulge, disk=disk)
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    Phase 2: Fit the lens`s `MassProfile``s and source galaxy`s light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy bulge+disk model from phase 1.
        2) Set priors on the `EllipticalIsothermal` (y,x) centre such that we assume the image is centred around the 
           lens galaxy`s bulge.
    """

    """Setup: Include an `ExternalShear` in the mass model if turned on in _SetupMass_. """

    if not setup.setup_mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = phase1.result.model.galaxies.lens.bulge.centre_0
    mass.centre_1 = phase1.result.model.galaxies.lens.bulge.centre_1

    phase2 = al.PhaseImaging(
        phase_name="phase_2__mass_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=phase1.result.instance.galaxies.lens.bulge,
                disk=phase1.result.instance.galaxies.lens.disk,
                mass=mass,
                shear=shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, bulge=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=60),
    )

    """
    Phase 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens`s bulge, disk, mass, and source model and priors using the results of phases 1 and 2.
    """

    phase3 = al.PhaseImaging(
        phase_name="phase_3__light_bulge_disk__mass_sie__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=phase1.result.model.galaxies.lens.bulge,
                disk=phase1.result.model.galaxies.lens.disk,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                bulge=phase2.result.model.galaxies.source.sersic,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    """
    Phase 4: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens`s bulge, disk and mass model to the results of phase 3.
    """

    phase4 = al.PhaseImaging(
        phase_name="phase_4__source_inversion_initialization",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=phase3.result.instance.galaxies.lens.bulge,
                disk=phase3.result.instance.galaxies.lens.disk,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.mass,
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
    Phase 5: Fit the lens`s bulge, disk and mass using the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the source `Inversion` parameters to the results of phase 4.
        2) Set priors on the lens galaxy bulge, disk and mass using the results of phase 3.
    """

    phase5 = al.PhaseImaging(
        phase_name="phase_5__light_bulge_disk__mass_sie__source_inversion",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=phase3.result.model.galaxies.lens.bulge,
                disk=phase3.result.model.galaxies.lens.disk,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4, phase5)
