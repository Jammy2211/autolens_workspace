from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autofit.tools import phase as autofit_ph
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
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
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens Mass constant from previous pipeline, Source Light constant from previous pipeline.
# Notes: Uses a 3D grid of subhalo (y,x) and mass, which is set via the config.

# Phase 2:

# Description: Perform the subhalo detection analysis.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens Mass constant from previous pipeline, source light variable from previous pipeline.
# Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 3:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens Mass variable from previous pipeline, source light and subhalo mass variable from phase 2.
# Notes: None

# Phase 4:

# Description: Change SIE mass profile to PowerLaw, to refine power-law slope.
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: None
# Prior Passing: Lens Mass, source light and subhalo mass variable from phase 3.
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations.

def make_pipeline(phase_folders=None, phase_tagging=True, sub_grid_size=2, bin_up_factor=None, positions_threshold=None,
                  inner_mask_radii=None, interp_pixel_scale=None):

    pipeline_name = 'pipeline_subhalo__lens_pl_shear_subhalo_source_inversion'

    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    ### Phase 2 ###

    # In phase 2, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model and source-pixelization parameters are held fixed to the best-fit values from phase 2.
    # 2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(autofit_ph.as_grid_search(ph.LensSourcePlanePhase)):

        @property
        def grid_priors(self):
            return [self.variable.subhalo.mass.centre_0, self.variable.subhalo.mass.centre_1]

        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_pl_shear_source_inversion').constant.len

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24)
            
            self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.0001, upper_limit=0.1)
            self.lens_galaxies.subhalo.mass.scale_radius = prior.UniformPrior(lower_limit=0.0, upper_limit=5.0)
            self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = \
                results.from_phase('phase_2_lens_pl_shear_refine_source_inversion').constant.source.pixelization

            self.source_galaxies.source.regularization = \
                results.from_phase('phase_2_lens_pl_shear_refine_source_inversion').variable.source.regularization

    phase2 = GridPhase(phase_name='phase_2_subhalo_search', phase_folders=phase_folders,
                       phase_tagging=phase_tagging,
                       lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw,
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

            ### Lens Mass, PL -> PL, Shear -> Shear ###
            
            self.lens_galaxies.lens = results.from_phase('phase_1_lens_pl_shear_source_inversion').variable.lens

            ### Subhalo, TruncatedNFW -> TruncatedNFW ###

            self.lens_galaxies.lens.subhalo = results.from_phase('phase_2_subhalo_search').best_result.variable.lens.subhalo

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = \
                results.from_phase('phase_2_lens_pl_shear_refine_source_inversion').constant.source.pixelization

            self.source_galaxies.source.regularization = \
                results.from_phase('phase_2_subhalo_search').best_result.variable.regularization

    phase3 = SubhaloPhase(phase_name='phase_3_subhalo_refine', phase_folders=phase_folders,
                          phase_tagging=phase_tagging,
                          lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw,
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