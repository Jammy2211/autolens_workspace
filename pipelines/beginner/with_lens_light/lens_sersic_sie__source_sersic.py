import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we fit the a strong lens using a Sersic light profile, SIE mass proflie and parametric Sersic
# source.

# The pipeline is three phases:

# Phase 1:

# Fit and subtract the lens light model.

# Lens Light: EllipticalSersic
# Lens Mass: None
# Source Light: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Fit the lens mass model and source light profile.

# Lens Light: EllipticalSersic
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: Lens Light (instance -> phase 1).
# Notes: Uses the lens subtracted image from phase 1.

# Phase 3:

# Refine the lens light and mass models and source light model using priors initialized from phases 1 and 2.

# Lens Light: EllipticalSersic
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: Lens light (model -> phase 1), lens mass and source light (model -> phase 2).
# Notes: None


def make_pipeline(
    setup,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__lens_sersic_sie__source_sersic"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    # This pipeline is tagged according to whether:

    # 1) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.source.tag_beginner + "__" + setup.mass.tag_beginner)

    ### SETUP SHEAR ###

    # Include the shear in the mass model if not switched off in the pipeline setup.

    if not setup.mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # In phase 1, we fit only the lens galaxy's light, where we:

    # 1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    sersic = af.PriorModel(al.lp.EllipticalSersic)
    sersic.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    sersic.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=redshift_lens, sersic=sersic)),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    # These lines customize MultiNest so that it samples non-linear parameter space faster. (Checkout
    # 'howtolens/chapter_"_lens_modeling/tutorial_7_multinest_black_magic' for details.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3
    phase1.optimizer.evidence_tolerance = evidence_tolerance

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and source galaxy's light, where we:

    # 1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
    # 2) Set priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for \
    #    the light profile in phase 1.
    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = phase1.result.model.galaxies.lens.sersic.centre_0
    mass.centre_1 = phase1.result.model.galaxies.lens.sersic.centre_1

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                sersic=phase1.result.instance.galaxies.lens.sersic,
                mass=mass,
                shear=shear,
            ),
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

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.2
    phase2.optimizer.evidence_tolerance = evidence_tolerance

    ### PHASE 3 ###

    # In phase 3, we fit simultaneously the lens and source galaxies, where we:

    # 1) Set the lens's light, mass, and source's light using the results of phases 1 and 2.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                sersic=phase1.result.model.galaxies.lens.sersic,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase2.result.model.galaxies.source.sersic,
            ),
        ),
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
