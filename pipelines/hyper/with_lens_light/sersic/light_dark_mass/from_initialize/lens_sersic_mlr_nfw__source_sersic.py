import autofit as af
import autolens as al

import os

# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using a parametric light profile, using a decomposed light and dark matter mass model including mass profile. The
# pipeline follows on from the initialize pipeline
# 'pipelines/with_lens_light/initialize/lens_sersic_sie__source_sersic_from_init.py'.

# The pipeline is one phase, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a decomposed light and dark matter profile, using a
#              parametric Sersic light profile for the source. The lens light model is fixed.
# Lens Light & Mass: EllipticalSersic
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: with_lens_light/initialize/lens_sersic_sie__source_sersic.py
# Prior Passing: None
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.


# Phase 2:

# Description: Fits the lens light and mass model as a decomposed light and dark matter profile, using a
#              parametric Sersic light profile for the source.
# Lens Light & Mass: EllipticalSersic
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: with_lens_light/initialize/lens_sersic_sie__source_sersic.py
# Prior Passing: None
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.


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
    pixel_scale_interpolation_grid=0.05,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_ldm__lens_sersic_mlr_nfw__source_sersic"
    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        include_shear=pipeline_settings.include_shear,
        align_light_dark_centre=pipeline_settings.align_light_dark_centre,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Fix the lens galaxy's light using the EllipticalSersic of the previous pipeline.
    # 2) Pass priors on the lens galaxy's SphericalNFW mass profile's centre using the EllipticalIsothermal fit of the
    #    previous pipeline, if the NFW centre is a free parameter.
    # 3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
    # 4) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        light=al.lmp.EllipticalSersic,
        dark=al.mp.SphericalNFW,
        shear=af.last.model.galaxies.lens.shear,
        hyper_galaxy=af.last.hyper_combined.instance.galaxies.lens.hyper_galaxy,
    )

    lens.light.centre = af.last.instance.galaxies.lens.light.centre
    lens.light.axis_ratio = af.last.instance.galaxies.lens.light.axis_ratio
    lens.light.phi = af.last.instance.galaxies.lens.light.phi
    lens.light.intensity = af.last.instance.galaxies.lens.light.intensity
    lens.light.effective_radius = af.last.instance.galaxies.lens.light.effective_radius
    lens.light.sersic_index = af.last.instance.galaxies.lens.light.sersic_index

    if pipeline_settings.align_light_dark_centre:

        lens.dark.centre = lens.light.centre

    elif not pipeline_settings.align_light_dark_centre:

        lens.dark.centre = af.last.model_absolute(a=0.05).galaxies.lens.light.centre

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic_mlr_nfw__source_sersic__fixed_lens_light",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                light=af.last.model.galaxies.source.light,
                hyper_galaxy=af.last.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=af.last.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.hyper_background_noise,
        sub_size=sub_size,
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

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Pass priors on the lens galaxy's light using the EllipticalSersic of the previous pipeline.
    # 2) Pass priors on the lens galaxy's SphericalNFW mass profile's centre using the EllipticalIsothermal fit of the
    #    previous pipeline, if the NFW centre is a free parameter.
    # 3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
    # 4) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sersic_mlr_nfw__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=phase1.result.model.galaxies.lens.light,
                dark=phase1.result.model.galaxies.lens.dark,
                shear=phase1.result.model.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                light=phase1.result.model.galaxies.source.light,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.hyper_background_noise,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 75
    phase2.optimizer.sampling_efficiency = 0.2

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
