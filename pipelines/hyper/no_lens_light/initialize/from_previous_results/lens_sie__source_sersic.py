import autofit as af
import autolens as al

# In this pipeline, we'll perform an initialize analysis which fits an image with a source galaxy and no lens light
# component. The pipeline is as follows:

# Phase 1:

# Description: initialize the lens mass model and source light profile.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: None


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
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_init_hyper__lens_sie__source_sersic"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        hyper_galaxies=pipeline_settings.hyper_galaxies,
        hyper_image_sky=pipeline_settings.hyper_image_sky,
        hyper_background_noise=pipeline_settings.hyper_background_noise,
        include_shear=pipeline_settings.include_shear,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### SETUP SHEAR ###

    # If the pipeline should include shear, add this class below so that it enters the phase.

    # After this pipeline this shear class is passed to all subsequent pipelines, such that the shear is either
    # included or omitted throughout the entire pipeline.

    if pipeline_settings.include_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    class ResultPhase(al.PhaseImaging):
        def modify_image(self, image, results):
            return image  # <- input function which creates LOS simulated image here.

        def customize_priors(self, results):

            ### Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.mass.centre = results.last.model_absolute(
                a=0.05
            ).galaxies.lens.mass.centre

            self.galaxies.lens.mass.axis_ratio = results.last.model_absolute(
                a=0.1
            ).galaxies.lens.mass.axis_ratio

            self.galaxies.lens.mass.phi = results.last.model_absolute(
                a=20.0
            ).galaxies.lens.mass.phi

            self.galaxies.lens.mass.einstein_radius = results.last.model_relative(
                r=0.05
            ).galaxies.lens.mass.einstein_radius

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear.magnitude = results.last.model.absolute(
                    a=0.02
                ).galaxies.lens.shear.magnitude

                self.galaxies.lens.shear.phi = results.last.model_absolute(
                    a=20.0
                ).galaxies.lens.shear.phi

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source.light.centre = results.last.model_absolute(
                a=0.1
            ).galaxies.source.light.centre

            self.galaxies.source.light.axis_ratio = results.last.model_absolute(
                a=0.1
            ).galaxies.source.light.axis_ratio

            self.galaxies.source.light.phi = results.last.model_absolute(
                a=20.0
            ).galaxies.source.light.phi

            self.galaxies.source.light.intensity = results.last.model_relative(
                r=0.3
            ).galaxies.source.light.intensity

            self.galaxies.source.light.effective_radius = results.last.model_relative(
                r=0.3
            ).galaxies.source.light.effective_radius

            self.galaxies.source.light.sersic_index = results.last.model_relative(
                r=0.3
            ).galaxies.source.light.sersic_index

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxies:

                self.galaxies.source.hyper_galaxy = (
                    results.last.hyper_combined.instance.galaxies.source.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.instance.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.instance.hyper_background_noise
                )

    phase1 = ResultPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=al.mp.EllipticalIsothermal, shear=shear
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    return al.PipelineDataset(pipeline_name, phase1, hyper_mode=True)
