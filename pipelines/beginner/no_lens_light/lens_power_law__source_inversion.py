import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we fit the a strong lens using a power-law mass proflie and a source which uses an
# inversion.

# The first 3 phases are identical to the pipeline 'lens_sie__source_inversion.py'.

# The pipeline is four phases:

# Phase 1:

# Fit the lens mass model and source light profile.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: None.
# Notes: None.

# Phase 2:

# Fit the source inversion using the lens mass profile inferred in phase 1.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Prior Passing: Lens & Mass (instance -> phase1).
# Notes: Lens mass fixed, source inversion parameters vary.

# Phase 3:

# Refines the lens light and SIE mass models using the source inversion of phase 2.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: pixelization (default=VoronoiMagnification) + regularization (default=Constant)
# Prior Passing: Lens Mass (model -> phase 1), Source Inversion (instance -> phase 2)
# Notes: Lens mass varies, source inversion parameters fixed.

# Phase 4:

# Fit the power-law mass model, using priors from the SIE mass model of phase 3 and a source inversion.

# Lens Mass: EllipticalPowerLaw + ExternalShear
# Source Light: pixelization (default=VoronoiMagnification) + regularization (default=Constant)
# Prior Passing: Lens Mass (model -> phase 3), Source Inversion (instance -> phase 3)
# Notes: Lens mass varies, source inversion parameters fixed.


def make_pipeline(
    setup,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__lens_power_law__source_inversion"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    # This pipeline is tagged according to whether:

    # 1) The lens galaxy mass model includes an external shear.
    # 2) The pixelization and regularization scheme of the pipeline (fitted in phases 3 & 4).

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.source.tag_beginner + "__" + setup.mass.tag_beginner)

    ### SETUP SHEAR ###

    # Include the shear in the mass model if not switched off in the pipeline setup.

    if not setup.mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and source light, where we:

    # 1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, mass=mass, shear=shear),
            source=al.GalaxyModel(
                redshift=redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    # These lines customize MultiNest so that it samples non-linear parameter space faster. (Checkout
    # 'howtolens/chapter_"_lens_modeling/tutorial_7_multinest_black_magic' for details.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = evidence_tolerance

    ### PHASE 2 ###

    #  In phase 2, we fit the input pipeline pixelization & regularization, where we:

    # 1) Set lens's mass model using the results of phase 1.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__source_inversion_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=setup.source.pixelization,
                regularization=setup.source.regularization,
            ),
        ),
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    ### PHASE 3 ###

    # In phase 3, we fit the lens's mass using the input pipeline pixelization & regularization, where we:

    # 1) Fix the source inversion parameters to the results of phase 2.
    # 2) Set priors on the lens galaxy mass using the results of phase 1.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__source_inversion",
        phase_folders=phase_folders,
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
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    # We now 'extend' phase 3 with an additional 'inversion phase' which uses the best-fit mass model of phase 3 above
    # to refine the it inversion used, by fitting only the pixelization & regularization parameters. This is equivalent
    # to phase 2 above, but makes the code shorter and more easy to read.

    # The the inversion phase results are accessible as attributes of the phase results and used in phase 4 below.

    phase3 = phase3.extend_with_inversion_phase()

    ### PHASE 4 ###

    # In phase 4, we fit the lens galaxy's mass with a power-law and a source inversion, where we:

    # 1) Use the source inversion of phase 3's extended inversion phase.
    # 2) Set priors on the lens galaxy mass using the EllipticalIsothermal and ExternalShear of phase 3.

    # Setup the power-law mass profile and initialize its priors from the SIE.

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = phase3.result.model.galaxies.lens.mass.centre
    mass.axis_ratio = phase3.result.model.galaxies.lens.mass.axis_ratio
    mass.phi = phase3.result.model.galaxies.lens.mass.phi
    mass.einstein_radius = phase3.result.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_power_law__source_inversion",
        phase_folders=phase_folders,
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
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 75
    phase4.optimizer.sampling_efficiency = 0.2

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
