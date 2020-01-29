import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we fit the a strong lens using a SIE mass proflie and a source which uses an
# inversion.

# The pipeline is three phases:

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

# Refines the lens light and mass models using the source inversion of phase 2.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Prior Passing: Lens Mass (model -> phase 1), Source Inversion (instance -> phase 2)
# Notes: Lens mass varies, source inversion parameters fixed.


def make_pipeline(
    pipeline_general_settings,
    pipeline_source_settings,
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

    pipeline_name = "pipeline__lens_sie__source_inversion"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    # This pipeline's name is tagged according to whether:

    # 1) The lens galaxy mass model includes an external shear.
    # 2) The pixelization and regularization scheme of the pipeline (fitted in phases 3 & 4).

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag + pipeline_source_settings.tag)

    ### SETUP SHEAR ###

    # Include the shear in the mass model includes shear if this pipeline setting is True.

    if pipeline_general_settings.with_shear:
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
                redshift=redshift_source, light=al.lp.EllipticalSersic
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
                pixelization=pipeline_source_settings.pixelization,
                regularization=pipeline_source_settings.regularization,
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

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)