import autofit as af
import autolens as al

"""
In this pipeline, we fit _Imaging_ of a strong lens system where:

 - The lens galaxy's _LightProfile_ is modeled as an _EllipticalSersic_ and _EllipticalExponential_.
 - The lens galaxy's stellar _MassProfile_ is fitted with the _EllipticalSersic_ and _EllipticalExponential_ of the 
      _LightProfile_, where they are converted to a stellar mass distribution via constant mass-to-light ratios.
 - The lens galaxy's dark matter _MassProfile_ is modeled as a _SphericalNFW_.
 - The source galaxy's _LightProfile_ is modeled as an _EllipticalSersic_.  

The pipeline is five phases:

Phase 1:

    Fit and subtract the lens light model.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source _LightProfile_, where the _LightProfile_ parameters of the lens's 
    _LightMassProfile_ are fixed to the results of phase 1.
    
    Lens Light: EllipticalSersic + EllipticalExponential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the lens subtracted image from phase 1.

Phase 3:

    Refine the lens light and mass models and source light model using priors initialized from phases 1 and 2.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> phase 1), lens mass and source light (model -> phase 2).
    Notes: None

Phase 4:

    Fit the source _Inversion_ using the lens light and _MassProfile_'s inferred in phase 3.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (instance -> phase3).
    Notes: Lens mass fixed, source _Inversion_ parameters vary.

Phase 5:

    Refines the lens light and mass models using the source _Inversion_ of phase 4.
    
    Lens Light: EllipticalSersic + EllipticalExpoonential
    Lens Mass: EllipticalSersic + EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Light & Mass (model -> phase 3), Source _Inversion_ (instance -> phase 4)
    Notes: Lens mass varies, source _Inversion_ parameters fixed.
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__light_bulge_disk__mass_mlr_dark__source_inversion"

    """
    This pipeline is tagged according to whether:

        1) The bulge + disk centres or elliptical_comps are aligned.
        2) The disk component of the lens light model is an _EllipticalExponential_ or _EllipticalSersic_ profile.
        3) The centres of the lens galaxy bulge and dark matter are aligned.
        4) If the bulge and disk share the same mass-to-light ratio or each is fitted independently.
        5) The lens galaxy mass model includes an  _ExternalShear_.
    """

    setup.folders.append(pipeline_name)
    setup.folders.append(setup.tag)

    """
    Phase 1; Fit only the lens galaxy's light, where we:

        1) Set priors on the bulge and disk's (y,x) centres such that we assume the image is centred around 
           the lens galaxy OR,
        2) Fix the lens light centre to the input value in _SetupLightBulgeDisk_.
    """

    bulge = af.PriorModel(al.lp.EllipticalSersic)

    """Setup: Set whether the disk is modeled as an _EllipticalSersic_ or _EllipticalExponential_."""

    if setup.setup_light.disk_as_sersic:
        disk = af.PriorModel(al.lp.EllipticalSersic)
    else:
        disk = af.PriorModel(al.lp.EllipticalExponential)

    """Setup: Set the alignment of the bulge and disk's centres and elliptical components."""

    if setup.setup_light.align_bulge_disk_centre:
        bulge.centre = disk.centre

    if setup.setup_light.align_bulge_disk_elliptical_comps:
        bulge.elliptical_comps = disk.elliptical_comps

    """Setup: Fix the bulge and disk centres to the input value in _SetupLight_ if input."""

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
    Phase 2: Fit the lens's _MassProfile_'s and source galaxy's light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy bulge+disk model from phase 1.
        2) Set priors on the centre of the lens galaxy's dark matter _MassProfile_ by linking them to those inferred 
           for the _LightProfile_ in phase 1.
        3) Use a _SphericalNFWMCRLudlow_ model for the dark matter which sets its scale radius via a mass-concentration
           relation and the lens and source redshifts.
    """

    bulge = af.PriorModel(al.lmp.EllipticalSersic)

    bulge.centre = phase1.result.instance.galaxies.lens.bulge.centre
    bulge.elliptical_comps = phase1.result.instance.galaxies.lens.bulge.elliptical_comps
    bulge.intensity = phase1.result.instance.galaxies.lens.bulge.intensity
    bulge.effective_radius = phase1.result.instance.galaxies.lens.bulge.effective_radius
    bulge.sersic_index = phase1.result.instance.galaxies.lens.bulge.sersic_index

    if setup.setup_light.disk_as_sersic:
        disk = af.PriorModel(al.lmp.EllipticalSersic)
    else:
        disk = af.PriorModel(al.lmp.EllipticalExponential)

    disk.centre = phase1.result.instance.galaxies.lens.disk.centre
    disk.elliptical_comps = phase1.result.instance.galaxies.lens.disk.elliptical_comps
    disk.intensity = phase1.result.instance.galaxies.lens.disk.intensity
    disk.effective_radius = phase1.result.instance.galaxies.lens.disk.effective_radius
    if setup.setup_light.disk_as_sersic:
        disk.sersic_index = phase1.result.instance.galaxies.lens.disk.sersic_index

    """SLaM: Set all the bulge and disk mass-to-light ratios to one another if input in _SetupMassLightDark_."""

    if setup.setup_mass.constant_mass_to_light_ratio:
        bulge.mass_to_light_ratio = disk.mass_to_light_ratio

    dark = af.PriorModel(al.mp.SphericalNFWMCRLudlow)

    """Setup: Align the centre of the bulge _LightProfile_ and dark matter _MassProfile_ if input in _SetupMassLightDark_."""

    if setup.setup_mass.align_light_bulge_centre:
        dark.centre = bulge.centre
    else:
        dark.centre = phase1.result.model.galaxies.lens.bulge.centre

    dark.mass_at_200 = af.LogUniformPrior(lower_limit=5e8, upper_limit=5e14)
    dark.redshift_object = setup.redshift_lens
    dark.setup.redshift_source = setup.redshift_source

    """Setup: Include an _ExternalShear_ in the mass model if turned on in _SetupMass_. """

    if not setup.setup_mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    phase2 = al.PhaseImaging(
        phase_name="phase_2__mass_mlr_dark__source_sersic__fixed_lens_light",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=bulge,
                disk=disk,
                dark=dark,
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

        1) Set the lens's bulge, disk, dark, and source's light using the results of phases 1 and 2.
    """

    bulge = af.PriorModel(al.lmp.EllipticalSersic)

    bulge.centre = phase1.result.model.galaxies.lens.bulge.centre
    bulge.elliptical_comps = phase1.result.model.galaxies.lens.bulge.elliptical_comps
    bulge.intensity = phase1.result.model.galaxies.lens.bulge.intensity
    bulge.effective_radius = phase1.result.model.galaxies.lens.bulge.effective_radius
    bulge.sersic_index = phase1.result.model.galaxies.lens.bulge.sersic_index
    bulge.mass_to_light_ratio = (
        phase2.result.model.galaxies.lens.bulge.mass_to_light_ratio
    )

    if setup.setup_light.disk_as_sersic:
        disk = af.PriorModel(al.lmp.EllipticalSersic)
        disk.sersic_index = phase1.result.model.galaxies.lens.disk.sersic_index
    else:
        disk = af.PriorModel(al.lmp.EllipticalExponential)

    disk.centre = phase1.result.model.galaxies.lens.disk.centre
    disk.elliptical_comps = phase1.result.model.galaxies.lens.disk.elliptical_comps
    disk.intensity = phase1.result.model.galaxies.lens.disk.intensity
    disk.effective_radius = phase1.result.model.galaxies.lens.disk.effective_radius
    disk.mass_to_light_ratio = (
        phase2.result.model.galaxies.lens.disk.mass_to_light_ratio
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__light_bulge_disk__mass_mlr_dark__source_sersic",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                bulge=bulge,
                disk=disk,
                dark=phase2.result.model.galaxies.lens.dark,
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
    Phase 4: Fit the input pipeline _Pixelization_ & _Regularization_, where we:

        1) Fix the lens's bulge, disk and mass model to the results of phase 3.
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
    Phase 5: Fit the lens's bulge, disk and mass using the input pipeline _Pixelization_ & _Regularization_, where we:

        1) Fix the source _Inversion_ parameters to the results of phase 4.
        2) Set priors on the lens galaxy bulge, disk and mass using the results of phase 3.
    """

    phase5 = al.PhaseImaging(
        phase_name="phase_5__light_bulge_disk__mass_mlr_dark__source_inversion",
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
