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

import os

# In this pipeline, we'll perform a subhalo analysis which determines the sensitivity map of a strong lens and 
# then attempts to detection subhalos by putting subhalos at fixed intevals on a 2D (y,x) grid. The lens includes a 
# light component and the source uses a light profile. The pipeline is as follows:

# Phase 1:

# Description: Perform the sensitivity analysis for subhalo locations.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens mass and source light (constant -> previous pipline).
# Notes: Uses the lens subtracted image of a previous pipeline. 
#        Uses a 3D grid of subhalo (y,x) and mass, which is set via the config.

# Phase 2:

# Description: Perform the subhalo detection analysis.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens mass (constant -> previous pipeline), source light (variable -> previous pipeline).
# Notes: Uses the lens subtracted image of a previous pipeline. 
#        Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 3:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens light and mass (variable -> previous pipeline), source light and subhalo mass (variable -> phase 2).
# Notes: None

def make_pipeline(
        phase_folders=None, tag_phases=True,
        redshift_lens=0.5, redshift_source=1.0,
        sub_grid_size=2, bin_up_factor=None, positions_threshold=None, inner_mask_radii=None, interp_pixel_scale=0.05):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pipeline_subhalo__lens_sersic_sie_shear_subhalo_source_sersic'
    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    # ### Phase 1 ###
    #
    # # In phase 1, we perform the sensitivity analysis of our lens, using a grid search of subhalo (y,x) coordinates and
    # # mass, where:
    #
    # # 1) The lens model and sourc light profile parameters are held fixed to the best-fit values from phase 2.
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
    #                    tag_phases=tag_phases,
    #                    lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalPowerLaw,
    #                                                           shear=mp.ExternalShear),
    #                                       subhalo=gm.GalaxyModel(mass=mp.SphericalNFW)),
    #                    source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
    #                    optimizer_class=nl.GridSearch)

    ### Phase 2 ###

    # In phase 2, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model parameters are held fixed to the best-fit values from phase 1 of the initialization pipeline.
    # 2) The source light-profile parameters are allowed to vary with customized priors.
    # 3) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 4) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(autofit_ph.as_grid_search(ph.LensSourcePlanePhase)):

        @property
        def grid_priors(self):
            return [self.variable.lens_galaxies.subhalo.mass.centre_0,
                    self.variable.lens_galaxies.subhalo.mass.centre_1]

        def pass_priors(self, results):

            ### Lens Light, Sersic -> Sersic ###

            self.lens_galaxies.lens.light = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                constant.lens_galaxies.lens.light

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                constant.lens_galaxies.lens

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24) ###

            self.lens_galaxies.subhalo.mass.kappa_s = prior.UniformPrior(lower_limit=0.0005, upper_limit=0.2)
            self.lens_galaxies.subhalo.mass.scale_radius = prior.UniformPrior(lower_limit=0.001, upper_limit=1.0)
            self.lens_galaxies.subhalo.mass.centre_0 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

            ### Source Light, Sersic -> Sersic ###

            self.source_galaxies.source.light.centre = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable_absolute(a=0.3).source_galaxies.source.light.centre

            self.source_galaxies.source.light.intensity = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable.source_galaxies.source.light.intensity

            self.source_galaxies.source.light.effective_radius = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable.source_galaxies.source.light.effective_radius

            self.source_galaxies.source.light.sersic_index = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable.source_galaxies.source.light.sersic_index

            self.source_galaxies.source.light.axis_ratio = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable.source_galaxies.source.light.axis_ratio

            self.source_galaxies.source.light.phi = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable.source_galaxies.source.light.phi

    phase2 = GridPhase(
        phase_name='phase_2_subhalo_search', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=redshift_lens, light=lp.EllipticalSersic,
                                               mass=mp.EllipticalPowerLaw, shear=mp.ExternalShear),
                           subhalo=gm.GalaxyModel(redshift=redshift_lens,mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, interp_pixel_scale=interp_pixel_scale,
        optimizer_class=nl.MultiNest, number_of_steps=4)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5

    class SubhaloPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Light, Sersic -> Sersic ###

            self.lens_galaxies.lens.light = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable.lens_galaxies.lens.light

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_1_lens_sersic_pl_shear_source_sersic').\
                variable.lens_galaxies.lens

            ### Subhalo, TruncatedNFW -> TruncatedNFW ###

            self.lens_galaxies.subhalo = results.from_phase('phase_2_subhalo_search').\
                best_result.variable.lens_galaxies.subhalo

            ### Source Light, Sersic -> Sersic ###

            self.source_galaxies.source = results.from_phase('phase_2_subhalo_search').\
                best_result.variable.source_galaxies.source

    phase3 = SubhaloPhase(
        phase_name='phase_3_subhalo_refine', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(lens=gm.GalaxyModel(redshift=redshift_lens, light=lp.EllipticalSersic,
                                               mass=mp.EllipticalPowerLaw, shear=mp.ExternalShear),
                           subhalo=gm.GalaxyModel(redshift=redshift_lens, mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, interp_pixel_scale=interp_pixel_scale,
        optimizer_class=nl.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 80
    phase3.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase2, phase3)