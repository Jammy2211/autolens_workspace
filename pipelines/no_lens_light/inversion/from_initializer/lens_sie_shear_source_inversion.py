from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform an inversion analysis which fits an image with a source galaxy and no lens light
# component. This reconstructs the source using a pxielized inversion, initialized using the light-profile source fit
# of a previous pipeline. The pipeline is as follows:

# Phase 1:

# Description: Initializes the inversion's pixelization and regularization, using a previous lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializers/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline).
# Notes: None

# Phase 2:

# Description: Refine the lens mass model using the source inversion.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializer/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable -> phase 1).
# Notes: Source inversion resolution is fixed.

# Phase 3:

# Description: Refine the source inversion using the new lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: None
# Prior Passing: Lens Mass (constant -> phase 2), source inversion (variable -> phase 1 & 2).
# Notes: Source inversion resolution varies.

def make_pipeline(phase_folders=None, phase_tagging=True, sub_grid_size=2, bin_up_factor=None, positions_threshold=None,
                  inner_mask_radii=None, interp_pixel_scale=None):

    pipeline_name = 'pipeline_inv__lens_sie_shear_source_inversion'

    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we initialize the inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 3 of the initializer pipeline.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_sie_shear_source_sersic').constant.lens

    phase1 = InversionPhase(phase_name='phase_1_initialize_inversion', phase_folders=phase_folders,
                            phase_tagging=phase_tagging,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor,
                            positions_threshold=positions_threshold, inner_mask_radii=inner_mask_radii,
                            interp_pixel_scale=interp_pixel_scale,
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

            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_sie_shear_source_sersic').variable.lens

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = \
                results.from_phase('phase_1_initialize_inversion').constant.source.pixelization

            self.source_galaxies.source.regularization = \
                results.from_phase('phase_1_initialize_inversion').variable.source.regularization

    phase2 = InversionPhase(phase_name='phase_2_lens_sie_shear_source_inversion', phase_folders=phase_folders,
                            phase_tagging=phase_tagging,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor,
                            positions_threshold=positions_threshold, inner_mask_radii=inner_mask_radii,
                            optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5

    ### PHASE 3 ###

    # In phase 3, we refine the inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 2.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_2_lens_sie_shear_source_inversion').constant.lens

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source = results.from_phase('phase_2_lens_sie_shear_source_inversion').variable.source

    phase3 = InversionPhase(phase_name='phase_3_lens_sie_shear_refine_source_inversion', phase_folders=phase_folders,
                            phase_tagging=phase_tagging,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor,
                            positions_threshold=positions_threshold, inner_mask_radii=inner_mask_radii,
                            interp_pixel_scale=interp_pixel_scale,
                            optimizer_class=nl.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 20
    phase3.optimizer.sampling_efficiency = 0.8

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)