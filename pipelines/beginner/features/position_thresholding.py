import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we use custom positions (generated with 'tools/data_making/imaging/positions_maker.py' file) in a
# pipeline. Phases are given a 'positions_threshold', which requires the positions drawn on the image to trace within
# this threshold for a given mass model. If they don't, the mass model is instantly discarded and resampled.

# This provides two benefits:

# 1) PyAutoLens does not waste computing time on mass models which will clearly give a poor fit to the dataset.

# 2) By discarding these mass models, non-linear parameter space will be less complex and thus easier to sample.

# Positions must be passed to the pipeline's 'run' function. See 'runners/features/position_thresholding.py'
# for an example of this.

# We use a simple two phase pipeline:

# Phase 1:

# Fit the lens and source model using positions to resample mass models.

# Lens Mass: EllipticalIsothermal
# Source Light: EllipticalSersic
# Prior Passing: None
# Notes: Use positions to resample mass models.

# Phase 1:

# Fit the lens and source model without using position thresholding.

# Lens Mass: EllipticalIsothermal
# Source Light: EllipticalSersic
# Prior Passing: Lens mass (model -> phase 1), source light (model -> phase 1)
# Notes: No position thresholding.


def make_pipeline(phase_folders=None, positions_threshold=None):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__feature"
    pipeline_tag = "position_thresholding"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    # A setup tag is automatically added to the phase path, making it clear the 'position_threshold' value used.
    # The positions_threshold_tag and phase name are shown for 3 example values:

    # positions_threshold=0.2 -> phase_path='phase_name/setup_pos_0.20'
    # positions_threshold=0.25, -> phase_path='phase_name/setup_pos_0.25'

    # If the positions_threshold is None, the tag is an empty string, thus not changing the phase name:

    # - positions_threshold=None, positions_threshold_tag='', phase_name=phase_name/setup

    ### PHASE 1 ###

    # In phase 1, we will:

    # 1) Input the positions_threshold such that mass models will be resampled (provided it is not None).

    phase1 = al.PhaseImaging(
        phase_name="phase_1__use_positions",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal),
            source=al.GalaxyModel(redshift=1.0, sersic=al.lp.EllipticalSersic),
        ),
        positions_threshold=0.3,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will:

    # 1) Set the priors on the lens galaxy's mass and source galaxy's light using those inferred in phase 1.
    # 2) Not specify a positions_threshold, so it is not used to resample mass models.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__no_positions",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.model.galaxies.lens.mass
            ),
            source=al.GalaxyModel(
                redshift=1.0, sersic=phase1.result.model.galaxies.source.sersic
            ),
        ),
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.2

    return al.PipelineDataset(pipeline_name, phase1, phase2)
