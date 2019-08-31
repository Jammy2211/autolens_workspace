import autofit as af
import autolens as al

# In this pipeline, we show how custom positions (generated using the tools/positions_maker.py file) can be input and
# used by a pipeline. Both phases will be input with a 'positions_threshold', which requires that a set of positions
# drawn on the image in the image plane trace within this threshold for a given mass model. If they don't, the mass
# model is instantly discarded and resampled. This provides two benefits:

# 1) PyAutoLens does not waste computing time on mass models which will clearly give a poor fit to the data.

# 2) By discarding these mass models, non-linear parameter space will be less complex and thus easier to sample.

# To use positions, the positions must be passed to the pipeline's 'run' function. See runners/runner_positions.py
# for an example of this.

# We use a simple two phase pipeline:

# Phase 1:

# Description: Fits the lens and source model using positions to resample mass models.
# Lens Mass: EllipitcalIsothermal
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Use positions to resample mass models.

# Phase 1:

# Description: Fits the lens and source model without using position thresholding.
# Lens Mass: EllipitcalIsothermal
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: No position thresholding.


def make_pipeline(phase_folders=None, positions_threshold=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_feature__position_thresholding"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings()

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    # A settings tag is automatically added to the phase path, making it clear the position threshold value used.
    # The positions_threshold_tag and phase name are shown for 3 example inner mask radii values:

    # positions_threshold=0.2 -> phase_path='phase_name/settings_pos_0.20'
    # positions_threshold=0.25, -> phase_path='phase_name/settings_pos_0.25'

    # If the positions_threshold is None, the tag is an empty string, thus not changing the phase name:

    # - positions_threshold=None, positions_threshold_tag='', phase_name=phase_name/settings

    ### PHASE 1 ###

    # In phase 1, we will:

    # 1) Input the positions_threshold, such that if the value is not None it is used to resample mass models.

    def mask_function(image):
        return al.Mask.circular(
            shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=2.5
        )

    phase1 = al.PhaseImaging(
        phase_name="phase_1__use_positions",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        positions_threshold=0.3,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will:

    # 1) Initialize the priors on the lens galaxy's mass and source galaxy's light by linking them to those inferred
    #    in phase 1.

    # 2) Not specify a positions_threshold, such that is defaults to None in the phase and is not used to resample
    #        mass models.

    class LensSubtractedPhase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens = results.from_phase(
                "phase_1__use_positions"
            ).variable.galaxies.lens

            self.galaxies.source = results.from_phase(
                "phase_1__use_positions"
            ).variable.galaxies.source

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__no_positions",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        positions_threshold=positions_threshold,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.2

    return al.PipelineImaging(pipeline_name, phase1, phase2)
