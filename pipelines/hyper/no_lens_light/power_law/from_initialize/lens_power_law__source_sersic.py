import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

# In this pipeline, we'll perform an analysis which fits an image with  no lens light, and a source galaxy using a
# parametric light profile, using a power-law mass profile. The pipeline follows on from the initialize pipeline
# ''pipelines/no_lens_light/initialize/lens_sie__source_sersic_from_init.py'.

# The pipeline is one phase, as follows:

# Phase 1:

# Description: Fits the lens mass model as a power-law, using a parametric Sersic light profile for the source.
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: no_lens_light/initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: None
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations.


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

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_power_law_hyper__lens_power_law__source_sersic"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
        hyper_galaxies=pipeline_settings.hyper_galaxies,
        hyper_image_sky=pipeline_settings.hyper_image_sky,
        hyper_background_noise=pipeline_settings.hyper_background_noise,
        include_shear=pipeline_settings.include_shear,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy mass using the EllipticalIsothermal fit of the previous pipeline, and
    #    source galaxy of the previous pipeline.

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ### Lens Mass, SIE -> PL ###

            self.galaxies.lens.mass.centre = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass.centre

            self.galaxies.lens.mass.axis_ratio = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass.axis_ratio

            self.galaxies.lens.mass.phi = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass.phi

            self.galaxies.lens.mass.einstein_radius = (
                results.from_phase("phase_1__lens_sie__source_sersic")
                .variable_absolute(a=0.3)
                .galaxies.lens.mass.einstein_radius
            )

            ### Lens Shear, Shear -> Shear ###

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_1__lens_sie__source_sersic"
                ).variable.galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source.light = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.source.light

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxies:

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

    phase1 = LensSourcePhase(
        phase_name="phase_1__lens_power_law__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalPowerLaw,
                shear=mp.ExternalShear,
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

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    return pipeline.PipelineImaging(pipeline_name, phase1, hyper_mode=True)
