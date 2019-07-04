import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

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

    pipeline_name = 'pl__position_thresholding'

    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    phase_folders.append(pipeline_name)

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
        return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=2.5)

    phase1 = phase_imaging.LensSourcePlanePhase(
        phase_name='phase_1_use_positions', phase_folders=phase_folders, tag_phases=True,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic)),
        mask_function=mask_function, positions_threshold=0.3,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will:

    # 1) Initialize the priors on the lens galaxy's mass and source galaxy's light by linking them to those inferred
    #    in phase 1.

    # 2) Not specify a positions_threshold, such that is defaults to None in the phase and is not used to resample
#        mass models.

    class LensSubtractedPhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_1_use_positions').\
                variable.lens_galaxies.lens

            self.source_galaxies.source = results.from_phase('phase_1_use_positions').\
                variable.source_galaxies.source

    phase2 = LensSubtractedPhase(
        phase_name='phase_2_no_positions', phase_folders=phase_folders, tag_phases=True,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic)),
        mask_function=mask_function, positions_threshold=positions_threshold,
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.2

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)