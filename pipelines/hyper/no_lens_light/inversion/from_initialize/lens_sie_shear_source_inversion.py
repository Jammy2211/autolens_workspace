import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging, phase_extensions
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform an inversion analysis which fits an image with a source galaxy and no lens light
# component.
#
# This first reconstructs the source using a magnification based pxielized inversion, initialized using the
# light-profile source fit of a previous pipeline. This ensures that the hyper-image used by the pl_pixelization and
# pl_regularization is fitted using a pixelized source-plane, which ensures that irregular structure in the lensed
# source is adapted too.

# The pipeline is as follows:

# Phase 1:

# Description: initialize the inversion's pixelization and regularization, using a magnification based pixel-grid and
#              the previous lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline).
# Notes: None

# Phase 2:

# Description: Refine the lens mass model using the source inversion.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable -> phase 1).
# Notes: Source inversion resolution is fixed.

# Phase 3:

# Description: initialize the inversion's pixelization and regularization, using the input hyper-pixelization,
#              hyper-regularization and the previous lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: pl_pixelization + pl_regularization
# Previous Pipelines: None
# Prior Passing: Lens Mass (constant -> phase 2), source inversion (variable -> phase 1 & 2).
# Notes: Source inversion resolution varies.

# Phase 4:

# Description: Refine the lens mass model using the hyper-inversion.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: pl_pixelization + pl_regularization
# Previous Pipelines: initialize/lens_sie_shear_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable -> phase 1).
# Notes: Source inversion is fixed.


def make_pipeline(
    pl_hyper_galaxies=True,
    pl_hyper_background_sky=True,
    pl_hyper_background_noise=True,
    pl_pixelization=pix.VoronoiBrightnessImage,
    pl_regularization=reg.AdaptiveBrightness,
    phase_folders=None,
    tag_phases=True,
    redshift_lens=0.5,
    redshift_source=1.0,
    sub_grid_size=2,
    bin_up_factor=None,
    positions_threshold=None,
    inner_mask_radii=None,
    interp_pixel_scale=None,
    inversion_pixel_limit=None,
    cluster_pixel_scale=0.1,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_inv__lens_sie_shear_source_inversion"

    pipeline_name = tag.pipeline_name_from_name_and_settings(
        pipeline_name=pipeline_name,
        pixelization=pl_pixelization,
        regularization=pl_regularization,
    )

    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we initialize a Voronoi Magnification inversion's resolution and Constant regularization scheme's
    # regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 3 of the initialize pipeline.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    # We begin with a Voronoi Magnification + Constant inversion because if our source is too complex to have its
    # morphology well fitted by the initialize pipeline's Sersic profile, the model image will be inadequate to use as
    # a hyper-image.

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase(
                "phase_1_lens_sie_shear_source_sersic"
            ).constant.lens_galaxies.lens

            ## Set all hyper-galaxies if feature is turned on ##

            if pl_hyper_background_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pl_hyper_background_noise:

                self.hyper_noise_background = (
                    results.last.hyper_combined.constant.hyper_noise_background
                )

    phase1 = InversionPhase(
        phase_name="phase_1_initialize_magnification_inversion",
        phase_folders=phase_folders,
        tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            )
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            )
        ),
        sub_grid_size=sub_grid_size,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pl_hyper_galaxies,
        include_background_sky=pl_hyper_background_sky,
        include_background_noise=pl_hyper_background_noise,
        inversion=False,
    )

    ### PHASE 2 ###

    # In phase 2, we fit the lens's mass and source galaxy using the magnification inversion, where we:

    # 1) Initialize the priors on the lens galaxy mass using the results of the previous pipelinen.
    # 2) Initialize the priors of all source inversion parameters from phase 1.

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase(
                "phase_1_lens_sie_shear_source_sersic"
            ).variable.lens_galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source = results.from_phase(
                "phase_1_initialize_magnification_inversion"
            ).constant.source_galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            if pl_hyper_background_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pl_hyper_background_noise:

                self.hyper_noise_background = (
                    results.last.hyper_combined.constant.hyper_noise_background
                )

    phase2 = InversionPhase(
        phase_name="phase_2_lens_sie_shear_source_magnification_inversion",
        phase_folders=phase_folders,
        tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            )
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            )
        ),
        sub_grid_size=sub_grid_size,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pl_hyper_galaxies,
        include_background_sky=pl_hyper_background_sky,
        include_background_noise=pl_hyper_background_noise,
        inversion=False,
    )

    ### PHASE 3 ###

    # In phase 3, we initialize the brightness adaption's inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 3 of the initialize pipeline.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase(
                "phase_2_lens_sie_shear_source_magnification_inversion"
            ).constant.lens_galaxies.lens

            ## Set all hyper-galaxies if feature is turned on ##

            if pl_hyper_background_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pl_hyper_background_noise:

                self.hyper_noise_background = (
                    results.last.hyper_combined.constant.hyper_noise_background
                )

    phase3 = InversionPhase(
        phase_name="phase_3_initialize_inversion",
        phase_folders=phase_folders,
        tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            )
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pl_pixelization,
                regularization=pl_regularization,
            )
        ),
        sub_grid_size=sub_grid_size,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 20
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=pl_hyper_galaxies,
        include_background_sky=pl_hyper_background_sky,
        include_background_noise=pl_hyper_background_noise,
        inversion=True,
    )

    ### PHASE 4 ###

    # In phase 4, we fit the lens's mass and source galaxy using a hyper inversion, where we:

    # 1) Initialize the priors on the lens galaxy mass using the results of phase 2.
    # 2) Initialize the priors of all source inversion parameters from phase 3.

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.lens_galaxies.lens = results.from_phase(
                "phase_2_lens_sie_shear_source_magnification_inversion"
            ).variable.lens_galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source = results.from_phase(
                "phase_3_initialize_inversion"
            ).hyper_combined.constant.source_galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            if pl_hyper_galaxies:

                self.source_galaxies.source.hyper_galaxy = (
                    results.last.hyper_combined.constant.source_galaxies.source.hyper_galaxy
                )

            if pl_hyper_background_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pl_hyper_background_noise:

                self.hyper_noise_background = (
                    results.last.hyper_combined.constant.hyper_noise_background
                )

    phase4 = InversionPhase(
        phase_name="phase_4_lens_sie_shear_source_inversion",
        phase_folders=phase_folders,
        tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            )
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pl_pixelization,
                regularization=pl_regularization,
            )
        ),
        sub_grid_size=sub_grid_size,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=pl_hyper_galaxies,
        include_background_sky=pl_hyper_background_sky,
        include_background_noise=pl_hyper_background_noise,
        inversion=True,
    )

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4)
