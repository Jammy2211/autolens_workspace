from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autofit.tools import phase as autofit_ph
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform a subhalo analysis which determines the sensitivity map of a strong lens and 
# then attempts to detection subhalos by putting subhalos at fixed intevals on a 2D (y,x) grid. The source uses an
# inversion. The pipeline is as follows:

# Phase 1:

# Description: Perform the sensitivity analysis for subhalo locations.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens light, mass and source light (constant -> previous pipline).
# Notes: Uses the lens subtracted image of a previous pipeline.
#        Uses a 3D grid of subhalo (y,x) and mass, which is set via the config.

# Phase 2:

# Description: Perform the subhalo detection analysis.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens light, mass (constant -> previous pipeline), source inversion (variable -> previous pipeline).
# Notes: Uses the lens subtracted image of a previous pipeline.
#        Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 3:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens light and mass (variable -> previous pipeline), source inversion and subhalo mass (variable -> phase 2).
# Notes: None

def make_pipeline(phase_folders=None, tag_phases=True, sub_grid_size=2, bin_up_factor=None, positions_threshold=None,
                  inner_mask_radii=None, interp_pixel_scale=None):

    pipeline_name = 'pipeline_subhalo__lens_sersic_sie_shear_subhalo_source_inversion'

    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    # ### Phase 1 ###
    #
    # # In phase 1, we perform the sensitivity analysis of our lens, using a grid search of subhalo (y,x) coordinates and
    # # mass, where:
    #
    # # 1) The lens model and source-pixelization parameters are held fixed to the best-fit values from phase 2.
    #
    # class GridPhase(ph.LensSourcePlanePhase):
    #
    #     def pass_priors(self, results):
    #
    #         self.lens_galaxies.lens.mass = results.from_phase('phase_1_source').constant.lens.mass
    #         self.source_galaxies.source = results.from_phase('phase_1_source').constant.source
    #
    #         self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    #         self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    #         self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.00001, upper_limit=0.002)
    #         self.lens_galaxies.subhalo.mass.scale_radius = 5.0
    #
    # phase2 = GridPhase(phase_name='phase_2_sensitivity', phase_folders=phase_folders,
    #                    lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw,
    #                                                           shear=mp.ExternalShear),
    #                                       subhalo=gm.GalaxyModel(mass=mp.SphericalNFW)),
    #                    source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
    #                                                                            regularization=reg.Constant)),
    #                    optimizer_class=nl.GridSearch)

    ### Phase 2 ###

    # In phase 2, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model parameters are held fixed to the best-fit values from phase 1 of the initialization pipeline.
    # 2) The source-pixelization resolution is fixed to the best-fit values the inversion initialized pipeline, the
    #    regularized coefficienct is free to vary.
    # 3) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 4) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(autofit_ph.as_grid_search(ph.LensSourcePlanePhase)):

        @property
        def grid_priors(self):
            return [self.variable.subhalo.mass.centre_0, self.variable.subhalo.mass.centre_1]

        def pass_priors(self, results):

            ### Lens Light, Sersic -> Sersic ###

            self.lens_galaxies.lens.light = \
                results.from_phase('phase_1_lens_sersic_pl_shear_source_inversion').constant.lens.light

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_sersic_pl_shear_source_inversion').constant.lens
            
            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24) ###
            
            self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.0001, upper_limit=0.1)
            self.lens_galaxies.subhalo.mass.scale_radius = prior.UniformPrior(lower_limit=0.0, upper_limit=5.0)
            self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            
            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = \
                results.from_phase('phase_2_lens_sersic_pl_shear_refine_source_inversion').constant.source.pixelization
            
            self.source_galaxies.source.regularization = \
                results.from_phase('phase_2_lens_sersic_pl_shear_refine_source_inversion').variable.source.regularization

    phase2 = GridPhase(phase_name='phase_2_subhalo_search', phase_folders=phase_folders,
                       tag_phases=tag_phases,
                       lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic, 
                                                              mass=mp.EllipticalPowerLaw,
                                                              shear=mp.ExternalShear),
                                          subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                       source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                  regularization=reg.Constant)),
                       optimizer_class=nl.MultiNest,
                       sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor,
                       positions_threshold=positions_threshold, inner_mask_radii=inner_mask_radii,
                       interp_pixel_scale=interp_pixel_scale,
                       number_of_steps=4)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.3

    class SubhaloPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):
            
            ### Lens Light, Sersic -> Sersic ###

            self.lens_galaxies.lens.light = \
                results.from_phase('phase_1_lens_sersic_pl_shear_source_inversion').variable.lens.light

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_sersic_pl_shear_source_inversion').variable.lens

            ### Subhalo, TruncatedNFW -> TruncatedNFW ###

            self.lens_galaxies.subhalo = results.from_phase('phase_2_subhalo_search').best_result.variable.subhalo

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = \
                results.from_phase('phase_1_lens_sersic_pl_shear_source_inversion').constant.source.pixelization
            
            self.source_galaxies.source.regularization = \
                results.from_phase('phase_2_subhalo_search').best_result.variable.regularization

    phase3 = SubhaloPhase(phase_name='phase_3_subhalo_refine', phase_folders=phase_folders,
                          tag_phases=tag_phases,
                          lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic, 
                                                                 mass=mp.EllipticalPowerLaw,
                                                                 shear=mp.ExternalShear),
                                             subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                          source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                     regularization=reg.Constant)),
                          optimizer_class=nl.MultiNest,
                          sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor,
                          positions_threshold=positions_threshold, inner_mask_radii=inner_mask_radii,
                          interp_pixel_scale=interp_pixel_scale)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 80
    phase3.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase2, phase3)