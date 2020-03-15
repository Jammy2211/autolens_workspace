import autofit as af
import autolens as al

### HYPER PIPELINE INTERFACE ###

# This pipeline uses PyAutoLens's hyper-features and descriptions of these features is given below. Hyper-mode itself
# is described fully in chapter 5 of the HowToLens lecture series and I recommend you follow those tutorials first to
# gain a clear understanding of hyper-mode.

# HYPER MODEL OBJECTS #

# Below you'll note the following three hyper-model objects:

# - hyper_galaxy - If used, the noise-map in the bright regions of the galaxy is scaled.
# - hyper_image_sky - If used, the background sky of the image being fitted is included as part of the model.
# - hyper_background_noise - If used, the background noise of the noise-map is included as part of the model.

# An example of these objects being used to make a phase is as follows:

# phase = al.PhaseImaging(
#     phase_name="phase___hyper_example",
#     phase_folders=phase_folders,
#     galaxies=dict(
#         lens=al.GalaxyModel(
#             redshift=redshift_lens,
#             hyper_galaxy=phase_last.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
#         ),
#         source=al.GalaxyModel(
#             redshift=redshift_source,
#         ),
#     ),
#     hyper_image_sky=phase_last.result.hyper_combined.instance.optional.hyper_image_sky,
#     hyper_background_noise=phase_last.result.hyper_combined.instance.optional.hyper_background_noise,
#     optimizer_class=af.MultiNest,
# )

# Above, we pass inferred hyper model components to the phase (the 'hyper_combined' attribute is described next).

# What does the 'optional' attribute mean? It means that the component is only passed if it is used. For example, if
# hyper_image_sky is turned off (by settting hyper_image_sky to False in the PipelineGeneralSettings), this model
# component will not be passed. That is, it is optional.


# HYPER PHASES #

# The hyper-galaxies, hyper-image_sky and hyper-background-noise all have non-linear parameters we need to fit for
# during our modeling.
#
# How do we fit for the hyper-parameters using our non-linear search (e.g. MultiNest)? Typically, we don't fit for them
# simultaneously with the lens and source models, as this creates an unnecessarily large parameter space which we'd
# fail to fit accurately and efficiently.

# Instead, we 'extend' phases with extra phases that specifically fit certain components of hyper-galaxy-model. You've
# hopefully already seen the following code, which optimizes just the parameters of an inversion (e.g. the
# pixelization and regularization):

# phase1 = phase1.extend_with_inversion_phase()

# Extending a phase with hyper phases is just as easy:

# phase = phase.extend_with_multiple_hyper_phases(
#     inversion=True,
#     hyper-galaxy=True,
#     include_background_sky=True,
#     include_background_noise=True,
# )

# This extends the phase with 3 additional phases which:

# 1) Fit the inversion parameters using the pixelization and regularization scheme that were used in the main phase.
#    (e.g. a brightness-based pixelization and adaptive regularization scheme). The best-fit lens and source
#    models are used. This is called the 'inversion' phase.

# 2) Simultaneously fit the hyper-galaxies, background sky and background noise hyper parameters using the best-fit
#    lens and source models from the main phase. This phase only scales the noise and the image. This is called
#    the 'hyper-galaxy' phase.

# 3) Fit all of the components above using Gaussian priors centred on the resulting values of phases 1) and 2). This is
#    important as there is a trade-off between increasing the noise in the lens / source and changing the pixelization /
#    regularization hyper-galaxy-parameters. This is called the 'hyper_combined' phase.

# Above, we used the results of the 'hyper_combined' phase to setup the hyper-galaxies, hyper_image_sky, and
# hyper_background_noise. Typically, we set these components up as 'instances' whose parameters are fixed during the
# main phases which fit the lens and source models.

### PIPELINE DESCRIPTION ###

# In this pipeline, we fit the a strong lens using a SIE mass proflie and a source which uses an inversion. The
# pipeline will use hyper-features, that adapt the inversion and other aspects of the model to the data being fitted.

# The pipeline is as follows:

# Phase 1:

# Fit the lens mass model and source light profile.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: None
# Notes: None

# Phase 2:

# Fit the inversion's pixelization and regularization, using a magnification
# based pixel-grid and the previous lens mass model.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Prior Passing: Lens Mass (instance -> phase 1).
# Notes: Lens mass fixed, source inversion parameters vary.

# Phase 3:

# Refine the lens mass model using the source inversion.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Prior Passing: Lens Mass (model -> previous pipeline), source inversion (instance -> phase 2).
# Notes: Lens mass varies, source inversion parameters fixed.

# Phase 4:

# Fit the inversion's pixelization and regularization, using the input pixelization,
# regularization and the previous lens mass model.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: setup.pixelization + setup.regularization
# Prior Passing: Lens Mass (instance -> phase 3).
# Notes:  Lens mass fixed, source inversion parameters vary.

# Phase 5:

# Refine the lens mass model using the inversion.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: setup.pixelization + setup.regularization
# Prior Passing: Lens Mass (model -> phase 3), source inversion (instance -> phase 4).
# Notes: Lens mass varies, source inversion parameters fixed.


def make_pipeline(
    setup,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__lens_sie__source_inversion"

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.
    # 3) The pixelization and regularization scheme of the pipeline (fitted in phases 4 & 5).

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.general.tag)
    phase_folders.append(setup.source.tag_beginner + setup.mass.tag_beginner)

    ### SETUP SHEAR ###

    # Include the shear in the mass model if not switched off in the pipeline setup.

    if not setup.mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and source galaxy.

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=al.mp.EllipticalIsothermal, shear=shear
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = evidence_tolerance

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    ### PHASE 2 ###

    # Phases 2 & 3 use a magnification based pixelization and constant regularization scheme to reconstruct the source.
    # The pixelization & regularization input via the pipeline setup are not used until phases 4 & 5.

    # This is because a pixelization / regularization that adapts to the source's surface brightness uses a previous
    # model image of that source (its 'hyper-image'). If the source's true morphology is irregular, or there are
    # multiple sources, the Sersic profile used in phase 1 would give a poor hyper-image. In contrast, the
    # inversion below will accurately capture such a source.

    # In phase 2, we fit the pixelization and regularization, where we:

    # 1) Fix the lens mass model to the mass-model inferred by the previous pipeline.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__source_inversion_magnification_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 20
    phase2.optimizer.sampling_efficiency = 0.8
    phase2.optimizer.evidence_tolerance = 0.1

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 3 ###

    # In phase 3, we fit the lens's mass and source galaxy using the magnification inversion, where we:

    # 1) Fix the source inversion parameters to the results of phase 2.
    # 2) Set priors on the lens galaxy mass using the results of the previous pipeline.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sie__source_inversion_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5
    phase3.optimizer.evidence_tolerance = evidence_tolerance

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 4 ###

    # In phase 4, we fit the input pipeline pixelization & regularization, where we:

    # 1) Fix the lens mass model to the mass-model inferred in phase 3.

    phase4 = al.PhaseImaging(
        phase_name="phase_4__source_inversion_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=setup.source.pixelization,
                regularization=setup.source.regularization,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8
    phase4.optimizer.evidence_tolerance = 0.1

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 5 ###

    # In phase 5, we fit the lens's mass using the input pipeline pixelization & regularization, where we:

    # 1) Fix the source inversion parameters to the results of phase 4.
    # 2) Set priors on the lens galaxy mass using the results of phase 3.

    phase5 = al.PhaseImaging(
        phase_name="phase_5__lens_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase4.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase4.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase4.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 50
    phase5.optimizer.sampling_efficiency = 0.5

    phase5 = phase5.extend_with_multiple_hyper_phases(
        inversion=True,
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4, phase5)
