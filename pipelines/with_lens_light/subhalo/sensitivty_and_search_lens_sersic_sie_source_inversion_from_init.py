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
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_source_inversion_from_pipeline.py
# Prior Passing: Lens light, mass and source light (constant -> previous pipline).
# Notes: Uses the lens subtracted image of a previous pipeline.
#        Uses a 3D grid of subhalo (y,x) and mass, which is set via the config.

# Phase 2:

# Description: Perform the subhalo detection analysis.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_source_inversion_from_pipeline.py
# Prior Passing: Lens light, mass (constant -> previous pipeline), source inversion (variable -> previous pipeline).
# Notes: Uses the lens subtracted image of a previous pipeline.
#        Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 3:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_source_inversion_from_pipeline.py
# Prior Passing: Lens light and mass (variable -> previous pipeline), source inversion and subhalo mass (variable -> phase 2).
# Notes: None

# Phase 4:

# Description: Change SIE mass profile to PowerLaw, to refine power-law slope.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: None
# Prior Passing: Lens light, mass, source inversion and subhalo mass (variable -> phase 3).
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations.

def make_pipeline(phase_folders=None, interp_pixel_scale=0.05, bin_up_factor=1):

    pipeline_name = 'pipeline_subhalo_sensitivity_and_search_lens_sersic_sie_source_inversion'

    bin_up_factor_tag = tag.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=bin_up_factor)
    interp_pixel_scale_tag = tag.interp_pixel_scale_tag_from_interp_pixel_scale(interp_pixel_scale=interp_pixel_scale)

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

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
    #                    lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
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

            self.lens_galaxies.lens.mass = results.from_phase('phase_2_inversion').constant.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_2_inversion').constant.lens.shear
            
            self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.0001, upper_limit=0.1)
            self.lens_galaxies.subhalo.mass.scale_radius = prior.UniformPrior(lower_limit=0.0, upper_limit=5.0)
            self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

            self.source_galaxies.source.pixelization = results.from_phase('phase_2_inversion').constant.source.pixelization
            self.source_galaxies.source.regularization = results.from_phase('phase_2_inversion').variable.source.regularization

    phase2 = GridPhase(phase_name='phase_2_subhalo_search', phase_folders=phase_folders,
                       phase_tag=bin_up_factor_tag,
                       lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                              shear=mp.ExternalShear),
                                          subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                       source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                  regularization=reg.Constant)),
                       bin_up_factor=bin_up_factor,
                       number_of_steps=4, optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.3

    class SubhaloPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):
            
            self.lens_galaxies.lens.mass = results.from_phase('phase_2_inversion').variable.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase('phase_2_inversion').variable.lens.shear
            self.lens_galaxies.lens.subhalo = results.from_phase('phase_2_subhalo_search').best_result.variable.lens.subhalo

            self.source_galaxies.source.pixelization = results.from_phase('phase_2_inversion').variable.source.pixelization
            self.source_galaxies.source.regularization = results.from_phase('phase_2_subhalo_search').best_result.variable.regularization

    phase3 = SubhaloPhase(phase_name='phase_3_subhalo_refine', phase_folders=phase_folders,
                          phase_tag=bin_up_factor_tag,
                          lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                 shear=mp.ExternalShear),
                                             subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                          source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                     regularization=reg.Constant)),
                          bin_up_factor=bin_up_factor,
                          optimizer_class=nl.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 80
    phase3.optimizer.sampling_efficiency = 0.3

    class SubhaloPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):
            
            self.lens_galaxies.lens.shear = results.from_phase('phase_3_subhalo_refine').variable.lens.shear
            self.lens_galaxies.subhalo.mass = results.from_phase('phase_3_subhalo_refine').variable.subhalo.mass

            self.source_galaxies.source.pixelization = results.from_phase('phase_3_subhalo_refine').variable.variable.pixelization
            self.source_galaxies.source.regularization = results.from_phase('phase_3_subhalo_refine').best_result.variable.regularization

            self.lens_galaxies.lens.mass.centre_0 = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.centre_0
            self.lens_galaxies.lens.mass.centre_1 = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.centre_1
            self.lens_galaxies.lens.mass.axis_ratio = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.axis_ratio
            self.lens_galaxies.lens.mass.phi = results.from_phase('phase_3_subhalo_refine').variable.lens.mass.phi
            einstein_radius_value = results.from_phase('phase_3_subhalo_refine').constant.lens.mass.einstein_radius
            self.lens_galaxies.lens.mass.einstein_radius = prior.GaussianPrior(mean=einstein_radius_value, sigma=0.2)

    phase4 = SubhaloPhase(phase_name='phase_4_power_law', phase_folders=phase_folders,
                          phase_tag=bin_up_factor_tag + interp_pixel_scale_tag,
                          lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw,
                                                                 shear=mp.ExternalShear),
                                             subhalo=gm.GalaxyModel(mass=mp.SphericalTruncatedNFWChallenge)),
                          source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                     regularization=reg.Constant)),
                          interp_pixel_scale=interp_pixel_scale, bin_up_factor=bin_up_factor,
                          optimizer_class=nl.MultiNest)

    phase4.optimizer.const_efficiency_mode = False
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_name, phase2, phase3, phase4)