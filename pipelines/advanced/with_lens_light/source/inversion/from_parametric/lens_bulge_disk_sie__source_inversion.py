import autofit as af
import autolens as al

# In this pipeline, we'll perform a source inversion analysis which fits an image with a source galaxy and a lens light
# component.

# Phases 1 & 2 first use a magnification based pixelization and constant regularization scheme to reconstruct the
# source (as opposed to immediately using the pixelization & regularization input via the pipeline settings).
# This ensures that if the input pixelization or regularization scheme uses hyper-images, they are initialized using
# a pixelized source-plane, which is key for lens's with multiple or irregular sources.

# The pipeline is as follows:

# Phase 1:

# Set inversion's pixelization and regularization, using a magnification
# based pixel-grid and the previous lens light and mass model.

# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: source/parametric/lens_bulge_disk_sie__source_sersic.py
# Prior Passing: Lens Light / Mass (instance -> previous pipeline).
# Notes: Lens light & mass fixed, source inversion parameters vary.

# Phase 2:

# Refine the lens mass model using the source inversion.

# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: source/parametric/lens_bulge_disk_sie__source_sersic.py
# Prior Passing: Lens Light & Mass (model -> previous pipeline), source inversion (instance -> phase 1).
# Notes: Lens light fixed, mass varies, source inversion parameters fixed.

# Phase 3:

# Fit the inversion's pixelization and regularization, using the input pixelization,
# regularization and the previous lens mass model.

# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: pipeline_settings.pixelization + pipeline_settings.regularization
# Previous Pipelines: None
# Prior Passing: Lens Light & Mass (instance -> phase 2).
# Notes:  Lens light & mass fixed, source inversion parameters vary.

# Phase 4:

# Refine the lens mass model using the inversion.

# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: pixelization + regularization
# Prior Passing: Lens Light & Mass (model -> phase 3), source inversion (instance -> phase 3).
# Notes: Lens light fixed, mass varies, source inversion parameters fixed.


def make_pipeline(
    pipeline_general_settings,
    pipeline_source_settings,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline_source__inversion__lens_bulge_disk_sie__source_inversion"

    # This pipeline's name is tagged according to whether:

    # 1) Hyper-fitting settings (galaxies, sky, background noise) are used.
    # 2) The pixelization and regularization scheme of the pipeline (fitted in phases 3 & 4).
    # 3) The lens light model is fixed during the analysis.
    # 4) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag + pipeline_source_settings.tag)

    ### PHASE 1 ###

    # In phase 1, we fit the pixelization and regularization, where we:

    # 1) Fix the lens light & mass model to the light & mass models inferred by the previous pipeline.

    phase1 = al.PhaseImaging(
        phase_name="phase_1__source_inversion_magnification_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last.instance.galaxies.lens.bulge,
                disk=af.last.instance.galaxies.lens.disk,
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
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_general_settings.hyper_galaxies,
        include_background_sky=pipeline_general_settings.hyper_image_sky,
        include_background_noise=pipeline_general_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 2 ###

    # In phase 2, we refine the len galaxy mass using an inversion. We will:

    # 1) Fix the source inversion parameters to the results of phase 1.
    # 2) Fix the lens light model to the results of the previous pipeline.
    # 3) Set priors on the lens galaxy mass from the previous pipeline.

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=af.last[-1].instance.galaxies.lens.bulge,
        disk=af.last[-1].instance.galaxies.lens.disk,
        mass=af.last[-1].model.galaxies.lens.mass,
        shear=af.last[-1].model.galaxies.lens.shear,
        hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_bulge_disk_sie__source_inversion_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 75
    phase2.optimizer.sampling_efficiency = 0.2
    phase2.optimizer.evidence_tolerance = evidence_tolerance

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_general_settings.hyper_galaxies,
        include_background_sky=pipeline_general_settings.hyper_image_sky,
        include_background_noise=pipeline_general_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 3 ###

    # In phase 3, we fit the input pipeline pixelization & regularization, where we:

    # 1) Fix the lens light model to the results of the previous pipeline.
    # 2) Fix the lens mass model to the mass-model inferred in phase 2.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__source_inversion_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase2.result.instance.galaxies.lens.bulge,
                disk=phase2.result.instance.galaxies.lens.disk,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=pipeline_source_settings.pixelization,
                regularization=pipeline_source_settings.regularization,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 20
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_general_settings.hyper_galaxies,
        include_background_sky=pipeline_general_settings.hyper_image_sky,
        include_background_noise=pipeline_general_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 4 ###

    # In phase 4, we fit the lens's mass using the input pipeline pixelization & regularization, where we:

    # 1) Fix the source inversion parameters to the results of phase 3.
    # 2) Fix the lens light model to the results of the previous pipeline.
    # 3) Set priors on the lens galaxy mass using the results of phase 2.

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=phase2.result.instance.galaxies.lens.bulge,
        disk=phase2.result.instance.galaxies.lens.disk,
        mass=phase2.result.model.galaxies.lens.mass,
        shear=phase2.result.model.galaxies.lens.shear,
        hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    # If the lens light is fixed, over-write the pass prior above to fix the lens light model.

    if pipeline_source_settings.fix_lens_light:

        lens.bulge = phase2.result.instance.galaxies.lens.bulge
        lens.disk = phase2.result.instance.galaxies.lens.disk

    # If they were aligned, unalign the lens light and mass given the model is now initialized.

    if (
        pipeline_source_settings.align_light_mass_centre
        or pipeline_source_settings.lens_mass_centre is not None
    ):

        lens.mass.centre = phase2.result.model_absolute(
            a=0.05
        ).galaxies.lens.bulge.centre

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_bulge_disk_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase3.result.instance.galaxies.source.pixelization,
                regularization=phase3.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 75
    phase4.optimizer.sampling_efficiency = 0.2
    phase4.optimizer.evidence_tolerance = evidence_tolerance

    phase4 = phase4.extend_with_multiple_hyper_phases(
        inversion=True,
        hyper_galaxy=pipeline_general_settings.hyper_galaxies,
        include_background_sky=pipeline_general_settings.hyper_image_sky,
        include_background_noise=pipeline_general_settings.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
