import autofit as af
import autolens as al

# In this pipeline, we'll perform a source inversion analysis which fits an image with a lens mass model and
# source galaxy.

# Phases 1 & 2 use a magnification based pixelization and constant regularization scheme to reconstruct the source
# (as opposed to immediately using the pixelization & regularization input via the pipeline setup). This ensures
# that if the input pixelization or regularization scheme use hyper-images, they are initialized using
# a pixelized source-plane, which is key for lens's with multiple or irregular sources.

# The pipeline is as follows:

# Phase 1:

# Fit the inversion's pixelization and regularization, using a magnification
# based pixel-grid and the previous lens mass model.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: source/parametric/lens_sie__source_sersic.py
# Prior Passing: Lens Mass (instance -> previous pipeline).
# Notes: Lens mass fixed, source inversion parameters vary.

# Phase 2:

# Refine the lens mass model using the source inversion.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: source/parametric/lens_sie__source_sersic.py
# Prior Passing: Lens Mass (model -> previous pipeline), source inversion (instance -> phase 1).
# Notes: Lens mass varies, source inversion parameters fixed.

# Phase 3:

# Fit the inversion's pixelization and regularization, using the input pixelization,
# regularization and the previous lens mass model.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: setup.pixelization + setup.regularization
# Previous Pipelines: None
# Prior Passing: Lens Mass (instance -> phase 2).
# Notes:  Lens mass fixed, source inversion parameters vary.

# Phase 4:

# Refine the lens mass model using the inversion.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: setup.pixelization + setup.regularization
# Previous Pipelines: source/parametric/lens_sie__source_sersic.py
# Prior Passing: Lens Mass (model -> phase 2), source inversion (instance -> phase 3).
# Notes: Lens mass varies, source inversion parameters fixed.


def make_pipeline(
    setup,
    real_space_mask,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline_source__inversion__lens_sie__source_inversion"

    # For pipeline tagging we need to set the source type
    setup.set_source_type(source_type=setup.source.inversion_tag)

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.
    # 3) The pixelization and regularization scheme of the pipeline (fitted in phases 3 & 4).

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.general.tag)
    phase_folders.append(setup.source.tag)

    ### PHASE 1 ###

    # In phase 1, we fit the pixelization and regularization, where we:

    # 1) Fix the lens mass model to the mass-model inferred by the previous pipeline.

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1__source_inversion_magnification_initialization",
        phase_folders=phase_folders,
        real_space_mask=real_space_mask,
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
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8
    phase1.optimizer.evidence_tolerance = 0.1

    phase1 = phase1.extend_with_multiple_hyper_phases(inversion=False)

    ### PHASE 2 ###

    # In phase 2, we fit the lens's mass and source galaxy using the magnification inversion, where we:

    # 1) Fix the source inversion parameters to the results of phase 1.
    # 2) Set priors on the lens galaxy mass using the results of the previous pipeline.

    phase2 = al.PhaseInterferometer(
        phase_name="phase_2__lens_sie__source_inversion_magnification",
        phase_folders=phase_folders,
        real_space_mask=real_space_mask,
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
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5
    phase2.optimizer.evidence_tolerance = evidence_tolerance

    phase2 = phase2.extend_with_multiple_hyper_phases(inversion=False)

    ### PHASE 3 ###

    # In phase 3, we fit the input pipeline pixelization & regularization, where we:

    # 1) Fix the lens mass model to the mass-model inferred in phase 2.

    phase3 = al.PhaseInterferometer(
        phase_name="phase_3__source_inversion_initialization",
        phase_folders=phase_folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=setup.source.pixelization,
                regularization=setup.source.regularization,
            ),
        ),
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 20
    phase3.optimizer.sampling_efficiency = 0.8
    phase3.optimizer.evidence_tolerance = 0.1

    phase3 = phase3.extend_with_multiple_hyper_phases(inversion=False)

    ### PHASE 4 ###

    # In phase 4, we fit the lens's mass using the input pipeline pixelization & regularization, where we:

    # 1) Fix the source inversion parameters to the results of phase 3.
    # 2) Set priors on the lens galaxy mass using the results of phase 2.

    phase4 = al.PhaseInterferometer(
        phase_name="phase_4__lens_sie__source_inversion",
        phase_folders=phase_folders,
        real_space_mask=real_space_mask,
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
            ),
        ),
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5
    phase4.optimizer.evidence_tolerance = evidence_tolerance

    phase4 = phase4.extend_with_multiple_hyper_phases(inversion=True)

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
