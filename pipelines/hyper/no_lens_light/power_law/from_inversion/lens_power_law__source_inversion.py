import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform an analysis which fits an image with no lens light, and a source galaxy using an
# inversion, using a power-law mass profile. The pipeline follows on from the inversion pipeline
# ''pipelines/no_lens_light/inversion/lens_sie__source_inversion_from_initialize.py'.

# The pipeline is two phases, as follows:

# Phase 1:

# Description: Fits the lens mass model as a power-law, using an inversion for the Source.
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: VoronoiBrightnessImage + Constant
# Previous Pipelines: no_lens_light/inversion/lens_sie__source_inversion_from_initialize.py
# Prior Passing: Lens Mass (variable -> previous pipeline), source inversion (variable / constant -> previous pipeline)
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
    use_inversion_border=True,
    inversion_pixel_limit=None,
    cluster_pixel_scale=0.1,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_power_law_hyper__lens_power_law__source_inversion"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
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

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy mass using the EllipticalIsothermal fit of the previous pipeline, and
    #    source inversion of the previous pipeline.

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ### Lens Mass, SIE -> PL ###

            self.galaxies.lens.mass.centre = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).variable.galaxies.lens.mass.centre

            self.galaxies.lens.mass.axis_ratio = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).variable.galaxies.lens.mass.axis_ratio

            self.galaxies.lens.mass.phi = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).variable.galaxies.lens.mass.phi

            self.galaxies.lens.mass.einstein_radius = (
                results.from_phase("phase_4__lens_sie__source_inversion")
                .variable_absolute(a=0.3)
                .galaxies.lens.mass.einstein_radius
            )

            ### Lens Shear, Shear -> Shear ###

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_4__lens_sie__source_inversion"
                ).variable.galaxies.lens.shear

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source.pixelization = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).hyper_combined.constant.galaxies.source.pixelization

            self.galaxies.source.regularization = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).hyper_combined.constant.galaxies.source.regularization

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
        phase_name="phase_1__lens_power_law__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalPowerLaw,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pipeline_settings.pixelization,
                regularization=pipeline_settings.regularization,
            ),
        ),
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

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=True,
    )

    return pipeline.PipelineImaging(pipeline_name, phase1, hyper_mode=True)
