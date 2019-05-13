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
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens Mass constant from previous pipeline, Source Light constant from previous pipeline.
# Notes: Uses a 3D grid of subhalo (y,x) and mass, which is set via the config.

# Phase 2:

# Description: Perform the subhalo detection analysis.
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens Mass constant from previous pipeline, source light variable from previous pipeline.
# Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 3:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_inversion_from_pipeline.py
# Prior Passing: Lens Mass variable from previous pipeline, source light and subhalo mass variable from phase 2.
# Notes: None

def make_pipeline(
        phase_folders=None, tag_phases=True,
        redshift_lens=0.5, redshift_source=1.0,
        sub_grid_size=2, bin_up_factor=None, positions_threshold=None, inner_mask_radii=None, interp_pixel_scale=0.05):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pipeline_subhalo__lens_pl_shear_subhalo_source_inversion'
    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

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
            return [self.variable.lens_galaxies.subhalo.mass.centre_0, self.variable.lens_galaxies.subhalo.mass.centre_1]

        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_pl_shear_source_inversion').\
                constant.lens_galaxies.lens

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24)
            
            self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.0005, upper_limit=0.2)
            self.lens_galaxies.subhalo.mass.scale_radius = prior.UniformPrior(lower_limit=0.001, upper_limit=1.0)
            self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = results.from_phase('phase_2_lens_pl_shear_refine_source_inversion').\
                constant.source_galaxies.source.pixelization

            self.source_galaxies.source.regularization = results.from_phase('phase_2_lens_pl_shear_refine_source_inversion').\
                variable.source_galaxies.source.regularization

    phase2 = GridPhase(
        phase_name='phase_2_subhalo_search', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=redshift_lens, mass=mp.EllipticalPowerLaw,
                                               shear=mp.ExternalShear),
                           subhalo=gm.GalaxyModel(redshift=redshift_lens, mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(source=gm.GalaxyModel(redshift=redshift_source, pixelization=pix.AdaptiveMagnification,
                                                  regularization=reg.Constant)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, nterp_pixel_scale=interp_pixel_scale,
        optimizer_class=nl.MultiNest, number_of_steps=4)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.3

    ### PHASE 3 ###

    # In phase 3, we refine the inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 2.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):
            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_2_subhalo_search'). \
                best_result.constant.lens_galaxies.lens

            self.lens_galaxies.subhalo = results.from_phase('phase_2_subhalo_search'). \
                best_result.constant.lens_galaxies.subhalo

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source = results.from_phase('phase_2_subhalo_search'). \
                best_result.variable.source_galaxies.source

    phase3 = InversionPhase(phase_name='phase_3_subhalo_refine_source_inversion', phase_folders=phase_folders,
                            tag_phases=tag_phases,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw,
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

    class SubhaloPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###
            
            self.lens_galaxies.lens = results.from_phase('phase_1_lens_pl_shear_source_inversion').\
                variable.lens_galaxies.lens

            ### Subhalo, TruncatedNFW -> TruncatedNFW ###

            self.lens_galaxies.subhalo.mass.centre = results.from_phase('phase_2_subhalo_search').\
                best_result.variable_absolute(a=0.3).lens_galaxies.subhalo.mass.centre

            self.lens_galaxies.subhalo.mass.kappa_s = results.from_phase('phase_2_subhalo_search').\
                best_result.variable_relative(r=0.5).lens_galaxies.subhalo.mass.kappa_s

            self.lens_galaxies.subhalo.mass.scale_radius = results.from_phase('phase_2_subhalo_search').\
                best_result.variable_relative(r=0.5).lens_galaxies.subhalo.mass.scale_radius

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source.pixelization = results.from_phase('phase_3_subhalo_refine_source_inversion').\
                constant.source_galaxies.source.pixelization

            self.source_galaxies.source.regularization = results.from_phase('phase_3_subhalo_refine_source_inversion').\
                best_result.variable.source_galaxies.source.regularization

    phase4 = SubhaloPhase(
        phase_name='phase_4_subhalo_refine', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=redshift_lens, mass=mp.EllipticalPowerLaw,
                                               shear=mp.ExternalShear),
                           subhalo=gm.GalaxyModel(redshift=redshift_lens, mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(source=gm.GalaxyModel(redshift=redshift_source, pixelization=pix.AdaptiveMagnification,
                                                   regularization=reg.Constant)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, interp_pixel_scale=interp_pixel_scale,
        optimizer_class=nl.MultiNest)

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 80
    phase4.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase2, phase3, phase4)