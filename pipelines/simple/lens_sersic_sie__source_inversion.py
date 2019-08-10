import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

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
    pixelization=pix.VoronoiMagnification,
    regularization=reg.Constant,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    sub_grid_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    positions_threshold=None,
    inner_mask_radii=None,
    interp_pixel_scale=None,
    use_inversion_border=True,
    inversion_pixel_limit=None,
    cluster_pixel_scale=0.1,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline__lens_sersic_sie__source_inversion"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
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
        shear = mp.ExternalShear
    else:
        shear = None

    # We will switch between a circular mask which includes the lens light and an annular mask which removes it.

    def mask_function_circular(image):
        return msk.Mask.circular(
            shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=3.0
        )

    def mask_function_annular(image):
        return msk.Mask.circular_annular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.3,
            outer_radius_arcsec=3.0,
        )

    ### PHASE 1 ###

    # In phase 1, we will fit only the lens galaxy's light, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    # 2) Use a circular mask which includes the lens and source galaxy light.

    class LensPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
            self.galaxies.lens.light.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensPhase(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(redshift=redshift_lens, light=lp.EllipticalSersic)
        ),
        mask_function=mask_function_circular,
        sub_grid_size=sub_grid_size,
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

    class LensSubtractedPhase(phase_imaging.PhaseImaging):
        def modify_image(self, image, results):
            return (
                image
                - results.from_phase(
                    "phase_1__lens_sersic"
                ).unmasked_lens_power_lawane_model_image
            )

        def pass_priors(self, results):

            self.galaxies.lens.mass.centre_0 = results.from_phase(
                "phase_1__lens_sersic"
            ).variable.galaxies.lens.light.centre_0

            self.galaxies.lens.mass.centre_1 = results.from_phase(
                "phase_1__lens_sersic"
            ).variable.galaxies.lens.light.centre_1

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens, mass=mp.EllipticalIsothermal, shear=shear
            ),
            source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic),
        ),
        mask_function=mask_function_annular,
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.2

    ### PHASE 3 ###

    # In phase 3, we will fit simultaneously the lens and source galaxies, where we:

    # 1) Initialize the lens's light, mass, shear and source's light using the results of phases 1 and 2.

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light, Sersic -> Sersic ###

            self.galaxies.lens.light = results.from_phase(
                "phase_1__lens_sersic"
            ).variable.galaxies.lens.light

            ## Lens Mass, SIE -> SIE ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass

            ## Lens Mass, Shear -> Shear ###

            if include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_2__lens_sie__source_sersic"
                ).variable.galaxies.lens.shear

            ### Source Inversion, Sersic -> Sersic ###

            self.galaxies.source = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.source

    phase3 = LensSourcePhase(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic),
        ),
        mask_function=mask_function_circular,
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
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

    class InversionPhase(phase_imaging.PhaseImaging):
        def modify_image(self, image, results):
            return (
                image
                - results.from_phase(
                    "phase_3__lens_sersic_sie__source_sersic"
                ).unmasked_lens_power_lawane_model_image
            )

        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).constant.galaxies.lens.mass

            ## Lens Mass, Shear -> Shear ###

            if include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_3__lens_sersic_sie__source_sersic"
                ).constant.galaxies.lens.shear

    phase4 = InversionPhase(
        phase_name="phase_4__source_inversion_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens, mass=mp.EllipticalIsothermal, shear=shear
            ),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pixelization,
                regularization=regularization,
            ),
        ),
        mask_function=mask_function_annular,
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        use_inversion_border=use_inversion_border,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
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

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light, Sersic -> Sersic ###

            self.galaxies.lens.light = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).variable.galaxies.lens.light

            ## Lens Mass, SIE -> SIE ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).variable.galaxies.lens.mass

            ## Lens Mass, Shear -> Shear ###

            if include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_3__lens_sersic_sie__source_sersic"
                ).variable.galaxies.lens.shear

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_4__source_inversion_initialization"
            ).inversion.constant.galaxies.source

    phase5 = InversionPhase(
        phase_name="phase_5_lens_sersic_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pixelization,
                regularization=regularization,
            ),
        ),
        mask_function=mask_function_circular,
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        use_inversion_border=use_inversion_border,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 60
    phase5.optimizer.sampling_efficiency = 0.4

    return pipeline.PipelineImaging(
        pipeline_name, phase1, phase2, phase3, phase4, phase5
    )
