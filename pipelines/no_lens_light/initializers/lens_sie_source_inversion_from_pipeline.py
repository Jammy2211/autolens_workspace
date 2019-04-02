from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform an initializer analysis which fits an image with a source galaxy and no lens light
# component. This reconstructs the source using a pxielized inversion, and uses the light-profile source fit of a
# previous pipeline. The pipeline is as follows:

# Phase 1:

# Description: Initializes the inversion's pixelization and regularization hyper-parameters, using a previous lens mass
#              model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline).
# Notes: None

# Phase 2:

# Description: Refine the lens mass model and source inversion.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable -> phase 1).
# Notes: None


def make_pipeline(phase_folders=None, bin_up_factor=1):

    pipeline_name = 'pipeline_init_lens_sie_source_inversion'

    bin_up_factor_tag = tag.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=bin_up_factor)

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we initialize the inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 3 of the initializer pipeline.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass = results.from_phase('phase_1_source').constant.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_1_source').constant.lens.shear

    phase1 = InversionPhase(phase_name='phase_1_inversion_init', phase_folders=phase_folders,
                            phase_tag=bin_up_factor_tag,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    ### PHASE 2 ###

    # In phase 2, we fit the lens's mass and source galaxy using an inversion, where we:

    # 1) Initialize the priors on the lens galaxy mass using the results of the previous pipeline.
    # 2) Initialize the priors of all source inversion parameters from phase 1.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass = results.from_phase('phase_1_source').variable.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_1_source').variable.lens.shear
            self.source_galaxies.source = results.from_phase('phase_1_inversion_init').variable.source

    phase2 = InversionPhase(phase_name='phase_2_inversion', phase_folders=phase_folders,
                            phase_tag=bin_up_factor_tag,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            bin_up_factor=bin_up_factor,
                            optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)