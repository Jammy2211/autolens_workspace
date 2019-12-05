import autofit as af
import autolens as al

# In this pipeline, we'll perform an inversion analysis which fits an image with a source galaxy and no lens light
# component.
#
# This first reconstructs the source using a magnification based pxielized inversion, initialized using the
# light-profile source fit of a previous pipeline. This ensures that the hyper_galaxies-image used by the pipeline_settings.pixelization and
# pipeline_settings.regularization is fitted using a pixelized source-plane, which ensures that irstructure in the lensed
# source is adapted too.

# The pipeline is as follows:

# Phase 1:

# Description: initialize the inversion's pixelization and regularization, using a magnification based pixel-grid and
#              the previous lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline).
# Notes: None

# Phase 2:

# Description: Refine the lens mass model using the source inversion.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable -> phase 1).
# Notes: Source inversion resolution is fixed.

# Phase 3:

# Description: initialize the inversion's pixelization and regularization, using the input hyper_galaxies-pixelization,
#              hyper_galaxies-regularization and the previous lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: pipeline_settings.pixelization + pipeline_settings.regularization
# Previous Pipelines: None
# Prior Passing: Lens Mass (constant -> phase 2), source inversion (variable -> phase 1 & 2).
# Notes: Source inversion resolution varies.

# Phase 4:

# Description: Refine the lens mass model using the hyper_galaxies-inversion.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: pipeline_settings.pixelization + pipeline_settings.regularization
# Previous Pipelines: initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable -> phase 1).
# Notes: Source inversion is fixed.


def make_pipeline(
    pipeline_settings,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    positions_threshold=None,
    inner_mask_radii=None,
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_inv_hyper__lens_sie__source_inversion"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        hyper_galaxies=pipeline_settings.hyper_galaxies,
        hyper_image_sky=pipeline_settings.hyper_image_sky,
        hyper_background_noise=pipeline_settings.hyper_background_noise,
        include_shear=pipeline_settings.include_shear,
        pixelization=pipeline_settings.pixelization,
        regularization=pipeline_settings.regularization,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we initialize a Voronoi Magnification inversion's resolution and Constant regularization scheme's
    # regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 3 of the initialize pipeline.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    # We begin with a Voronoi Magnification + Constant inversion because if our source is too complex to have its
    # morphology well fitted by the initialize pipeline's Sersic profile, the model image will be inadequate to use as
    # a hyper_galaxies-image.

    phase1 = al.PhaseImaging(
        phase_name="phase_1__source_inversion_magnification_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=af.last.instance.galaxies.lens.mass,
                shear=af.last.instance.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 2 ###

    # In phase 2, we fit the lens's mass and source galaxy using the magnification inversion, where we:

    # 1) Initialize the priors on the lens galaxy mass using the results of the previous pipelinen.
    # 2) Initialize the priors of all source inversion parameters from phase 1.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_inversion_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 3 ###

    # In phase 3, we initialize the brightness adaption's inversion's resolution and regularization coefficient, where we:

    # 1) Fix our mass model to the lens galaxy mass-model from phase 3 of the initialize pipeline.
    # 2) Use a circular mask which includes all of the source-galaxy light.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__source_inversion_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=pipeline_settings.pixelization,
                regularization=pipeline_settings.regularization,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 20
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 4 ###

    # In phase 4, we fit the lens's mass and source galaxy using a hyper_galaxies inversion, where we:

    # 1) Initialize the priors on the lens galaxy mass using the results of phase 2.
    # 2) Initialize the priors of all source inversion parameters from phase 3.

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase3.result.instance.galaxies.source.pixelization,
                regularization=phase3.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=True,
    )

    return al.PipelineDataset(
        pipeline_name, phase1, phase2, phase3, phase4, hyper_mode=True
    )
