import autofit as af
import autolens as al

# In this pipeline, we'll perform an analysis which initialize a lens model (the lens's light, mass and source's \
# light) and then fits the source galaxy using an inversion. This pipeline uses four phases:

# Phase 1:

# Description: initialize the lens light model to subtract the foreground lens
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Description: initialize the lens mass model and source light profile.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses the lens subtracted image from phase 1.

# Phase 3:

# Description: Refine the lens light and mass models and source light model.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens light (variable -> phase 1), lens mass and source light (variable -> phase 2).
# Notes: None

# Phase 4:

# Description: initialize the source inversion parameters.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline).
# Notes: None

# Phase 5:

# Description: Refines the lens light and mass models using the source inversion.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), Source Inversion (constant -> phase 1)
# Notes: None

# ***NOTE***
#
# Performing this analysis in a pipeline composed of 5 consectutive phases it not ideal, and it is better to
# breaking the pipeline down into multiple pipelines. This is what is done in the 'pipelines/with_lens_light'
# folder, using the pipelines:

# 1) initialize/lens_sersic_sie__source_sersic_from_init.py (phases 1->3)
# 2) initialize/lens_sersic_sie__source_inversion_from_pipeline.py (phases 4->5)

# See runners/runner_adding_pipelines.py for more details on adding pipelines.


def make_pipeline(
    include_shear=True,
    pixelization=al.pix.VoronoiMagnification,
    regularization=al.reg.Constant,
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

    pipeline_name = "pipeline__lens_sersic_sie__source_inversion"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        include_shear=include_shear,
        pixelization=pixelization,
        regularization=regularization,
    )

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### SETUP SHEAR ###

    # If the pipeline should include shear, add this class below so that it enters the phase.

    # After this pipeline this shear class is passed to all subsequent pipelines, such that the shear is either
    # included or omitted throughout the entire pipeline.

    if include_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    # We will switch between a circular mask which includes the lens light and an annular mask which removes it.

    def mask_function_circular(shape_2d, pixel_scales):
        return al.mask.circular(
            shape_2d=shape_2d, pixel_scales=pixel_scales, radius=3.0
        )

    def mask_function_annular(shape_2d, pixel_scales):
        return al.mask.circular_annular(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            inner_radius=0.3,
            outer_radius=3.0,
        )

    ### PHASE 1 ###

    # In phase 1, we will fit only the lens galaxy's light, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    # 2) Use a circular mask which includes the lens and source galaxy light.

    light = af.PriorModel(al.lp.EllipticalSersic)
    light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    light.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=redshift_lens, light=light)),
        mask_function=mask_function_circular,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout the tutorial '' in howtolens/chapter_2).

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and source galaxy's light, where we:

    # 1) Use a lens-subtracted image generated by subtracting model lens galaxy image from phase 1.
    # 2) Use a circular annular mask which includes only the source-galaxy light.
    # 3) Initialize the priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for \
    #    its light profile in phase 1.

    class LensSubtractedPhase(al.PhaseImaging):
        def modify_image(self, image, results):
            return image - phase1.result.unmasked_model_visibilities

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = phase1.result.model.galaxies.lens.light.centre_0
    mass.centre_1 = phase1.result.model.galaxies.lens.light.centre_1

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, mass=mass, shear=shear),
            source=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
        mask_function=mask_function_annular,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.2

    ### PHASE 3 ###

    # In phase 3, we will fit simultaneously the lens and source galaxies, where we:

    # 1) Initialize the lens's light, mass, shear and source's light using the results of phases 1 and 2.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=phase1.result.model.galaxies.lens.light,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                light=phase1.result.model.galaxies.source.light,
            ),
        ),
        mask_function=mask_function_circular,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    ### PHASE 4 ###

    # In phase 4, we initialize the inversion's resolution and regularization coefficient, where we:

    # 1) Use a lens-subtracted image generated by subtracting model lens galaxy image from phase 1.
    # 2) Fix our mass model to the lens galaxy mass-model from phase 2.
    # 3) Use a circular annular mask which includes only the source-galaxy light.

    class InversionPhase(al.PhaseImaging):
        def modify_image(self, image, results):
            return image - phase1.result.unmasked_model_visibilities

    phase4 = InversionPhase(
        phase_name="phase_4__source_inversion_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.mass,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=pixelization,
                regularization=regularization,
            ),
        ),
        mask_function=mask_function_annular,
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
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    phase4 = phase4.extend_with_inversion_phase()

    ### PHASE 5 ###

    # In phase 5, we fit the len galaxy light, mass and source galxy simultaneously, using an inversion. We will:

    # 1) Initialize the priors of the lens galaxy and source galaxy from phases 3+4.
    # 2) Use a circular mask which includes the lens and source galaxy light.

    phase5 = al.PhaseImaging(
        phase_name="phase_5__lens_sersic_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=phase3.result.model.galaxies.lens.light,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase4.result.model.galaxies.source.pixelization,
                regularization=phase4.result.model.galaxies.source.regularization,
            ),
        ),
        mask_function=mask_function_circular,
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

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 60
    phase5.optimizer.sampling_efficiency = 0.4

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4, phase5)
