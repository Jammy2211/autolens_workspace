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

# In this pipeline, we'll perform an analysis which fits an image with no lens light, and a source galaxy using an
# inversion, using a power-law mass profile. The pipeline follows on from the inversion pipeline
# ''pipelines/no_lens_light/inversion/lens_sie_shear_source_inversion_from_initializer.py'.

# The pipeline is two phases, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a power-law, using an inversion for the Source.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: no_lens_light/inversion/lens_sie_shear_source_inversion_from_initializer.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable / constant -> previous pipeline)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations.

# Phase 2:

# Description: Refines the inversion parameters, using a fixed mass model from phase 1.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: None
# Prior Passing: Lens Mass (constant -> phase 1), source inversion (variable -> phase 1)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations.

def make_pipeline(
        phase_folders=None, tag_phases=True,
        redshift_lens=0.5, redshift_source=1.0,
        sub_grid_size=2, bin_up_factor=None, positions_threshold=None, inner_mask_radii=None, interp_pixel_scale=0.05):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pipeline_pl__lens_sersic_pl_shear_source_inversion'
    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy mass using the EllipticalIsothermal fit of the previous pipeline, and
    #    source inversion of the previous pipeline.

    class LensSourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Light, Sersic -> Sersic ###

            self.lens_galaxies.lens.light = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                variable.lens.lens_galaxies.light

            ### Lens Mass, SIE -> Powerlaw ###

            self.lens_galaxies.lens.mass.centre = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                variable_absolute(a=0.05).lens_galaxies.lens.mass.centre

            self.lens_galaxies.lens.mass.axis_ratio = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                variable.lens_galaxies.lens.mass.axis_ratio

            self.lens_galaxies.lens.mass.phi = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                variable.lens_galaxies.lens.mass.phi

            self.lens_galaxies.lens.mass.einstein_radius_in_units = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                variable_absolute(a=0.3).lens_galaxies.lens.mass.einstein_radius_in_units

            ### Lens Shear, Shear -> Shear ###

            self.lens_galaxies.lens.mass.shear = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                variable.lens_galaxies.lens.shear

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = results.from_phase('phase_3_lens_sie_shear_refine_source_inversion').\
                constant.source_galaxies.source.pixelization

            self.source_galaxies.source.regularization = results.from_phase('phase_3_lens_sie_shear_refine_source_inversion').\
                variable.source_galaxies.source.regularization
            
    phase1 = LensSourcePhase(phase_name='phase_1_lens_sersic_pl_shear_source_inversion', phase_folders=phase_folders,
                             tag_phases=tag_phases,
                             lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                    mass=mp.EllipticalPowerLaw,
                                                                    shear=mp.ExternalShear)),
                             source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                             sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor,
                             positions_threshold=positions_threshold, inner_mask_radii=inner_mask_radii,
                             interp_pixel_scale=interp_pixel_scale,
                             optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 3 ###

    # In phase 3, we refine the inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 2.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Light & Mass, Sersic -> Sersic, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_sersic_pl_shear_source_inversion').\
                constant.lens_galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source = results.from_phase('phase_1_lens_sersic_pl_shear_source_inversion').\
                variable.source_galaxies.source

    phase2 = InversionPhase(phase_name='phase_2_lens_sersic_pl_shear_refine_source_inversion', phase_folders=phase_folders,
                            tag_phases=tag_phases,
                            lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                   mass=mp.EllipticalPowerLaw,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor,
                            positions_threshold=positions_threshold, inner_mask_radii=inner_mask_radii,
                            interp_pixel_scale=interp_pixel_scale,
                            optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)