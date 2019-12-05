import autofit as af
import autolens as al

# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using a parametric light profile, using a power-law mass profile. The pipeline follows on from the initialize pipeline
# ''pipelines/no_lens_light/initialize/lens_sersic_ie_source_sersic_from_init.py'.

# Alignment of the centre, phi and axis-ratio of the light profile's EllipticalSersic and EllipticalExponential
# profiles use the alignment specified in the previous pipeline.

# The pipeline is one phase, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as power-law profile, using a parametric Sersic light profile for the source.
# Lens Light & Mass: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipticalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: with_lens_light/initialize/lens_bulge_disk_sie__source_sersic.py
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
    pixel_scale_interpolation_grid=None,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_power_law__lens_bulge_disk_power_law__source_sersic"
    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        include_shear=pipeline_settings.include_shear,
        fix_lens_light=pipeline_settings.fix_lens_light,
        align_bulge_disk_centre=pipeline_settings.align_bulge_disk_centre,
        align_bulge_disk_axis_ratio=pipeline_settings.align_bulge_disk_axis_ratio,
        disk_as_sersic=pipeline_settings.disk_as_sersic,
        align_bulge_disk_phi=pipeline_settings.align_bulge_disk_phi,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Pass priors on the lens galaxy's light using the EllipticalSersic of the previous pipeline.
    # 2) Pass priors on the lens galaxy's  mass using the EllipticalIsothermal and ExternalShear fit of the previous
    #    pipeline.
    # 3) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = af.last.model.galaxies.lens.mass.centre
    mass.axis_ratio = af.last.model.galaxies.lens.mass.axis_ratio
    mass.phi = af.last.model.galaxies.lens.mass.phi
    mass.einstein_radius = af.last.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_bulge_disk_power_law__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last.model.galaxies.lens.bulge,
                disk=af.last.model.galaxies.lens.disk,
                mass=mass,
                shear=af.last.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, light=af.last.model.galaxies.source.light
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
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    return al.PipelineDataset(pipeline_name, phase1)
