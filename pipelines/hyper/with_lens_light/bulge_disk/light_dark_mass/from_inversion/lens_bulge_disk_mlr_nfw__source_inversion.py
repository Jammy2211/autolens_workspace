import autofit as af
import autolens as al


# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using an inversion, using a decomposed light and dark matter profile. The pipeline follows on from the
# inversion pipeline 'pipelines/with_lens_light/inversion/from_initialize/lens_bulge_disk_sie__source_inversion.py'.

# Alignment of the centre, phi and axis-ratio of the light profile's EllipticalSersic and EllipticalExponential
# profiles use the alignment specified in the previous pipeline.

# The pipeline is two phases, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a decomposed profile, using an inversion for the Source.
# Lens Light & Mass: EllipticalSersic + EllipticalExponential
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: with_lens_light/bulge_disk/inversion/from_initialize/lens_bulge_disk_sieexp_source_inversion.py
# Prior Passing: Lens Light (variable -> previous pipeline), Lens Mass (default),
#                Source Inversion (variable / constant -> previous pipeline)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.

# Phase 2:

# Description: Refines the inversion parameters, using a fixed mass model from phase 1.
# Lens Light & Mass: EllipticalSersic
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: None
# Prior Passing: Lens Light & Mass (constant -> phase 1), source inversion (variable -> phase 1)
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
    inversion_uses_border=True,
    inversion_pixel_limit=None,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_ldm__lens_bulge_disk_mlr_nfw__source_inversion"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        include_shear=pipeline_settings.include_shear,
        align_bulge_disk_centre=pipeline_settings.align_bulge_disk_centre,
        align_bulge_disk_phi=pipeline_settings.align_bulge_disk_phi,
        align_bulge_disk_axis_ratio=pipeline_settings.align_bulge_disk_axis_ratio,
        disk_as_sersic=pipeline_settings.disk_as_sersic,
        align_bulge_dark_centre=pipeline_settings.align_bulge_dark_centre,
        pixelization=pipeline_settings.pixelization,
        regularization=pipeline_settings.regularization,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Fix the lens galaxy's light using the EllipticalSersic and EllipticalExponential of the previous
    #    pipeline. This includes using the bulge-disk alignment assumed in that pipeline.
    # 2) Pass priors on the lens galaxy's SphericalNFW mass profile's centre using the EllipticalIsothermal fit of the
    #    previous pipeline, if the NFW centre is a free parameter.
    # 3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
    # 4) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.

    if pipeline_settings.disk_as_sersic:
        disk = af.PriorModel(al.lmp.EllipticalSersic)
    else:
        disk = af.PriorModel(al.lmp.EllipticalExponential)

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lmp.EllipticalSersic,
        disk=disk,
        dark=al.mp.SphericalNFW,
        shear=af.last.model.galaxies.lens.shear,
        hyper_galaxy=af.last.hyper_combined.instance.galaxies.lens.hyper_galaxy,
    )

    lens.bulge.centre = af.last.instance.galaxies.lens.bulge.centre
    lens.bulge.axis_ratio = af.last.instance.galaxies.lens.bulge.axis_ratio
    lens.bulge.phi = af.last.instance.galaxies.lens.bulge.phi
    lens.bulge.intensity = af.last.instance.galaxies.lens.bulge.intensity
    lens.bulge.effective_radius = af.last.instance.galaxies.lens.bulge.effective_radius
    lens.bulge.sersic_index = af.last.instance.galaxies.lens.bulge.sersic_index

    lens.disk.centre = af.last.instance.galaxies.lens.disk.centre
    lens.disk.axis_ratio = af.last.instance.galaxies.lens.disk.axis_ratio
    lens.disk.phi = af.last.instance.galaxies.lens.disk.phi
    lens.disk.intensity = af.last.instance.galaxies.lens.disk.intensity
    lens.disk.effective_radius = af.last.instance.galaxies.lens.disk.effective_radius

    if pipeline_settings.disk_as_sersic:
        lens.disk.sersic_index = af.last.instance.galaxies.lens.disk.sersic_index

    if pipeline_settings.align_bulge_dark_centre:

        lens.dark.centre = lens.bulge.centre

    elif not pipeline_settings.align_bulge_dark_centre:

        lens.dark.centre = af.last.model_absolute(a=0.05).galaxies.lens.bulge.centre

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_bulge_disk_mlr_nfw__source_inversion__fixed_lens_light",
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
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's light and mass and one source galaxy using the results of phase 1 as
    # initialization

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_bulge_disk_mlr_nfw__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last[-1].model.galaxies.lens.bulge,
                disk=af.last[-1].model.galaxies.lens.disk,
                dark=phase1.result.model.galaxies.lens.dark,
                shear=phase1.result.model.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
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
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 75
    phase2.optimizer.sampling_efficiency = 0.2

    phase2 = phase2.extend_with_inversion_phase()

    return al.PipelineDataset(pipeline_name, phase1, phase2)
