import autofit as af
import autolens as al

import os

# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using a parametric light profile, using a decomposed light and dark matter mass model including mass profile. The
# pipeline follows on from the initialize pipeline
# 'pipelines/with_lens_light/initialize/lens_bulge_disk_sie__source_sersic.py'.

# Alignment of the centre, phi and axis-ratio of the light profile's EllipticalSersic and EllipticalExponential
# profiles use the alignment specified in the previous pipeline.

# The pipeline is one phase, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a decomposed light and dark matter profile, using a
#              parametric Sersic light profile for the source.
# Lens Light & Mass: EllipticalSersic + EllipticalExponential
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: with_lens_light/sersic/initialize/lens_bulge_disk_sie__source_sersic.py
# Prior Passing: None
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.


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
    pixel_scale_interpolation_grid=0.05,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_ldm_hyper__lens_bulge_disk_mlr_nfw__source_sersic"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        hyper_galaxies=pipeline_settings.hyper_galaxies,
        hyper_image_sky=pipeline_settings.hyper_image_sky,
        hyper_background_noise=pipeline_settings.hyper_background_noise,
        include_shear=pipeline_settings.include_shear,
        align_bulge_disk_centre=pipeline_settings.align_bulge_disk_centre,
        align_bulge_disk_phi=pipeline_settings.align_bulge_disk_phi,
        align_bulge_disk_axis_ratio=pipeline_settings.align_bulge_disk_axis_ratio,
        disk_as_sersic=pipeline_settings.disk_as_sersic,
        align_bulge_dark_centre=pipeline_settings.align_bulge_dark_centre,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Pass priors on the lens galaxy's light using the EllipticalSersic and EllipticalExponential of the previous
    #    pipeline. This includes using the bulge-disk alignment assumed in that pipeline.
    # 2) Pass priors on the lens galaxy's SphericalNFW mass profile's centre using the EllipticalIsothermal fit of the
    #    previous pipeline, if the NFW centre is a free parameter.
    # 3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
    # 4) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.

    class LensSourcePhase(al.PhaseImaging):
        def customize_priors(self, results):

            ### Lens Light to Light + Mass, Sersic -> Sersic ###

            self.galaxies.lens.bulge_mass.centre = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.bulge.centre

            self.galaxies.lens.bulge_mass.axis_ratio = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.bulge.axis_ratio

            self.galaxies.lens.bulge_mass.phi = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.bulge.phi

            self.galaxies.lens.bulge_mass.intensity = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.bulge.intensity

            self.galaxies.lens.bulge_mass.effective_radius = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.bulge.effective_radius

            self.galaxies.lens.bulge_mass.sersic_index = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.bulge.sersic_index

            ### Lens Light to Light + Mass, Bulge -> Bulge, Disk -> Disk ###

            self.galaxies.lens.disk_mass.centre = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.disk.centre

            self.galaxies.lens.disk_mass.axis_ratio = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.disk.axis_ratio

            self.galaxies.lens.disk_mass.phi = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.disk.phi

            self.galaxies.lens.disk_mass.intensity = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.disk.intensity

            self.galaxies.lens.disk_mass.effective_radius = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.disk.effective_radius

            ### Lens Mass, SIE ->  NFW ###

            if pipeline_settings.align_bulge_dark_centre:

                self.galaxies.lens.dark.centre = self.galaxies.lens.bulge_mass.centre

            elif not pipeline_settings.align_bulge_dark_centre:

                self.galaxies.lens.dark.centre = (
                    results.from_phase("phase_4__lens_bulge_disk_sie__source_sersic")
                    .variable_absolute(a=0.05)
                    .galaxies.lens.bulge_mass.centre
                )

            ### Lens Shear, Shear -> Shear ###

            self.galaxies.lens.shear = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.source

            ## Set all hyper_galaxies-galaxies if feature is turned on ##

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

    phase1 = LensSourcePhase(
        phase_name="phase_1__lens_bulge_disk_mlr_nfw__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge_mass=al.light_and_mass_profiles.EllipticalSersic,
                disk_mass=al.light_and_mass_profiles.EllipticalExponential,
                dark=al.mass_profiles.SphericalNFW,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, light=al.light_profiles.EllipticalSersic
            ),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
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

    return al.PipelineImaging(pipeline_name, phase1, hyper_mode=True)
