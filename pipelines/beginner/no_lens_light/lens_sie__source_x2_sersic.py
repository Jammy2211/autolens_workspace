import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we fit the a strong lens using a SIE mass proflie and a source which uses two Sersic profiles.

# The pipeline is two phases:

# Phase 1:

# Fit the lens mass model and source light profile using x1 Sersic.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: None
# Notes: None

# Phase 2:

# Fit the lens mass model and source light profile using x1 source galaxies.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Galaxy 1 - Light: EllipticalSersic
# Source Galaxy 2 - Light: EllipticalSersic
# Prior Passing: Lens mass (model -> phase 1), Source Galaxy 1 Light (model -> phase 1)
# Notes: None


def make_pipeline(
    setup,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    auto_positions_factor=None,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__lens_sie__source_x2_sersic"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    # This pipeline is tagged according to whether:

    # 1) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(
        setup.source.tag_beginner_no_inversion + "__" + setup.mass.tag_beginner
    )

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
        phase_name="phase_1__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, mass=mass, shear=shear),
            source_0=al.GalaxyModel(
                redshift=redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        positions_threshold=positions_threshold,
        auto_positions_factor=auto_positions_factor,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=af.MultiNest,
    )

    # These lines customize MultiNest so that it samples non-linear parameter space faster. (Checkout
    # 'howtolens/chapter_"_lens_modeling/tutorial_7_multinest_black_magic' for details.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = evidence_tolerance

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Set the priors on the lens galaxy mass using the results of phase 1.
    # 2) Set the priors on the first source galaxy's light using the results of phase 1.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_x2_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source_0=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase1.result.model.galaxies.source_0.sersic,
            ),
            source_1=al.GalaxyModel(
                redshift=redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        positions_threshold=positions_threshold,
        auto_positions_factor=auto_positions_factor,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        non_linear_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2)
