import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging, phase_extensions
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform a subhalo analysis which determines the sensitivity map of a strong lens and 
# then attempts to detection subhalos by putting subhalos at fixed intevals on a 2D (y,x) grid. The source uses a
# light profile. The pipeline is as follows:

# Phase 1:

# Description: Perform the subhalo detection analysis.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens mass (constant -> previous pipeline), source light (variable -> previous pipeline).
# Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 2:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initializers/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens mass (variable -> previous pipeline), source light and subhalo mass (variable -> phase 2).
# Notes: None

def make_pipeline(
        phase_folders=None, tag_phases=True,
        pl_pixelization=pix.VoronoiMagnification, pl_regularization=reg.AdaptiveBrightness,
        redshift_lens=0.5, redshift_source=1.0,
        sub_grid_size=2, bin_up_factor=None, positions_threshold=None, inner_mask_radii=None, interp_pixel_scale=None,
        inversion_pixel_limit=None, cluster_pixel_scale=0.1):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pipeline_subhalo__lens_sie_shear_subhalo_source_inversion'

    pipeline_name = tag.pipeline_name_from_name_and_settings(
        pipeline_name=pipeline_name, pixelization=pl_pixelization, regularization=pl_regularization)

    phase_folders.append(pipeline_name)
    
    ### Phase 1 ###

    # In phase 1, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model and source-pixelization parameters are held fixed to the best-fit values from phase 1 of the 
    #    initialization pipeline.
    # 2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(af.phase.as_grid_search(phase_imaging.LensSourcePlanePhase)):

        @property
        def grid_priors(self):
            return [self.variable.lens_galaxies.subhalo.mass.centre_0, self.variable.lens_galaxies.subhalo.mass.centre_1]

        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                constant.lens_galaxies.lens

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24)

            self.lens_galaxies.subhalo.mass.kappa_s = af.prior.UniformPrior(lower_limit=0.0005, upper_limit=0.2)
            self.lens_galaxies.subhalo.mass.scale_radius = af.prior.UniformPrior(lower_limit=0.001, upper_limit=1.0)
            self.lens_galaxies.subhalo.mass.centre_0 = af.prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
            self.lens_galaxies.subhalo.mass.centre_1 = af.prior.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source = results.from_phase('phase_2_lens_sie_shear_source_inversion').inversion.\
                constant.source_galaxies.source

    phase1 = GridPhase(
        phase_name='phase_1_subhalo_search', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear),
            subhalo=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pl_pixelization,
                regularization=pl_regularization)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, interp_pixel_scale=interp_pixel_scale,
        inversion_pixel_limit=inversion_pixel_limit, cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest, number_of_steps=4)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.5

    class SubhaloPhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase('phase_2_lens_sie_shear_source_inversion').\
                variable.lens_galaxies.lens

            ### Subhalo, TruncatedNFW -> TruncatedNFW ###

            self.lens_galaxies.subhalo.mass.centre = results.from_phase('phase_1_subhalo_search').\
                best_result.variable_absolute(a=0.3).lens_galaxies.subhalo.mass.centre

            self.lens_galaxies.subhalo.mass.kappa_s = results.from_phase('phase_1_subhalo_search').\
                best_result.variable_relative(r=0.5).lens_galaxies.subhalo.mass.kappa_s

            self.lens_galaxies.subhalo.mass.scale_radius = results.from_phase('phase_1_subhalo_search').\
                best_result.variable_relative(r=0.5).lens_galaxies.subhalo.mass.scale_radius

            ### Source Light, Sersic -> Sersic ###

            self.source_galaxies.source = results.from_phase('phase_2_lens_sie_shear_source_inversion').inversion.\
                constant.source_galaxies.source

    phase2 = SubhaloPhase(
        phase_name='phase_2_subhalo_refine', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear),
            subhalo=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.SphericalTruncatedNFWChallenge)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pl_pixelization,
                regularization=pl_regularization)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, interp_pixel_scale=interp_pixel_scale,
        inversion_pixel_limit=inversion_pixel_limit, cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 80
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)