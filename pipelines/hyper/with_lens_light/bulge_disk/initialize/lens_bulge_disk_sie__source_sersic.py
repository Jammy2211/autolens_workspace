import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

# In this pipeline, we'll perform a basic analysis which initialize a lens model (the lens's light, mass and source's \
# light) and then fits the source galaxy using an inversion. This pipeline uses three phases:

# Phase 1:

# Description: initialize and subtracts the lens light model.
# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: None
# Source Light: None
# Previous Pipelines: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Description: initialize the lens mass model and source light profile, using the lens subtracted image from phase 1.
# Lens Light: None
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses the lens light subtracted image from phase 1

# Phase 3:

# Description: Refit the lens light models using a mass model and source light profile fixed from phase 2.
# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: lens mass and source (constant -> phase 2)
# Notes: None

# Phase 4:

# Description: Refine the lens light and mass models and source light profile, using priors from the previous 2 phases.
# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens light (variable -> phase 1), lens mass and source (variable -> phase 2)
# Notes: None


def make_pipeline(
    pipeline_settings,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    sub_grid_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    positions_threshold=None,
    inner_mask_radii=None,
    interp_pixel_scale=None,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is tagged according to
    # whether certain components of the lens light's bulge-disk model are aligned.

    pipeline_name = "pipeline_init_hyper__lens_bulge_disk_sie__source_sersic"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
        hyper_galaxies=pipeline_settings.hyper_galaxies,
        hyper_image_sky=pipeline_settings.hyper_image_sky,
        hyper_background_noise=pipeline_settings.hyper_background_noise,
        include_shear=pipeline_settings.include_shear,
        align_bulge_disk_centre=pipeline_settings.align_bulge_disk_centre,
        align_bulge_disk_phi=pipeline_settings.align_bulge_disk_phi,
        align_bulge_disk_axis_ratio=pipeline_settings.align_bulge_disk_axis_ratio,
        disk_as_sersic=pipeline_settings.disk_as_sersic,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### SETUP SHEAR ###

    # If the pipeline should include shear, add this class below so that it enters the phase.

    # After this pipeline this shear class is passed to all subsequent pipelines, such that the shear is either
    # included or omitted throughout the entire pipeline.

    if pipeline_settings.include_shear:
        shear = mp.ExternalShear
    else:
        shear = None

    ### SETUP DISK ###

    # Determine whether the profile of the disk in the pipeline is an EllipitcalExponential (Default) or
    # EllipticalSersic.

    # After this pipeline this class for the disk is passed to all subsequent pipelines, such that the disk is always
    # the same profile chosen in this pipeline.

    if pipeline_settings.disk_as_sersic:
        disk = lp.EllipticalSersic
    else:
        disk = lp.EllipticalExponential

    ### PHASE 1 ###

    # In phase 1, we will fit only the lens galaxy's light, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    # 2) Use a circular mask which includes the lens and source galaxy light.

    class BulgeDiskPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.bulge.centre = self.galaxies.lens.disk.centre

    phase1 = BulgeDiskPhase(
        phase_name="phase_1__lens_bulge_disk",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens, bulge=lp.EllipticalSersic, disk=disk
            )
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout the tutorial '' in howtolens/chapter_2).

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.3

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and source galaxy's light, where we:

    # 1) Use a lens-subtracted image generated by subtracting model lens galaxy image from phase 1.
    # 2) Initialize the priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for \
    #    the bulge of the light profile in phase 1.
    # 3) Have the option to use an annular mask removing the central light, if the inner_mask_radii parametr is input.

    class LensSubtractedPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light Bulge -> Bulge, Disk -> Disk ##

            self.galaxies.lens.bulge = results.from_phase(
                "phase_1__lens_bulge_disk"
            ).constant.galaxies.lens.bulge

            self.galaxies.lens.disk = results.from_phase(
                "phase_1__lens_bulge_disk"
            ).constant.galaxies.lens.disk

            ## Lens Mass, Move centre priors to centre of lens light ###

            self.galaxies.lens.mass.centre = (
                results.from_phase("phase_1__lens_bulge_disk")
                .variable_absolute(a=0.1)
                .galaxies.lens.bulge.centre
            )

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge=lp.EllipticalSersic,
                disk=disk,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 3 ###

    # In phase 3, we will refit the lens galaxy's light using a fixed mass andn source model above, where we:

    # 1) Do not use priors from phase 1 to initialize the lens's light, assuming the source light may of impacted them.
    # 2) Use a circular mask, to fully capture the lens and source light.

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            if pipeline_settings.align_bulge_disk_centre:
                self.galaxies.lens.bulge.centre = self.galaxies.lens.disk.centre

            if pipeline_settings.align_bulge_disk_axis_ratio:
                self.galaxies.lens.bulge.axis_ratio = self.galaxies.lens.disk.axis_ratio

            if pipeline_settings.align_bulge_disk_phi:
                self.galaxies.lens.bulge.phi = self.galaxies.lens.disk.phi

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).constant.galaxies.lens.mass

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_2__lens_sie__source_sersic"
                ).constant.galaxies.lens.shear

            ### Source Light, Bulge -> Bulge, Disk -> Disk ###

            self.galaxies.source.light = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).constant.galaxies.source.light

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

                self.galaxies.source.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase3 = LensSourcePhase(
        phase_name="phase_3__lens_bulge_disk_sie__source_fixed",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge=lp.EllipticalSersic,
                disk=disk,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 4 ###

    # In phase 4, we will fit simultaneously the lens and source galaxies, where we:

    # 1) Initialize the lens's light, mass, shear and source's light using the results of phases 1 and 2.
    # 2) Use a circular mask, to fully capture the lens and source light.

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Light, Sersic -> Sersic ###

            self.galaxies.lens.bulge = results.from_phase(
                "phase_3__lens_bulge_disk_sie__source_fixed"
            ).variable.galaxies.lens.bulge

            self.galaxies.lens.disk = results.from_phase(
                "phase_3__lens_bulge_disk_sie__source_fixed"
            ).variable.galaxies.lens.disk

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_2__lens_sie__source_sersic"
                ).variable.galaxies.lens.shear

            ### Source Light, Bulge -> Bulge, Disk -> Disk ###

            self.galaxies.source = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.source

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase4 = LensSourcePhase(
        phase_name="phase_4__lens_bulge_disk_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge=lp.EllipticalSersic,
                disk=disk,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 75
    phase4.optimizer.sampling_efficiency = 0.3

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    return pipeline.PipelineImaging(
        pipeline_name, phase1, phase2, phase3, phase4, hyper_mode=True
    )
