import autofit as af
import autolens as al

# In this pipeline, we fit the mass of a strong lens using a power-law + shear model.

# The mass model and source are initialized using an already run 'source' pipeline.

# The pipeline is one phases:

# Phase 1:

# Fit the lens mass model as a power-law, using the source model from a previous pipeline.
# Lens Mass: EllipticalPowerLaw + ExternalShear
# Source Light: Previous Pipeline Source.
# Previous Pipeline: no_lens_light/source/*/lens_sie__source_*py
# Prior Passing: Lens Mass (model -> previous pipeline), source (model / instance -> previous pipeline)
# Notes: If the source is parametric, its parameters are varied, if its an inversion, they are fixed.


def make_pipeline(
    pipeline_general_settings,
    pipeline_mass_settings,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    # A source tag distinguishes if the previous pipeline models used a parametric or inversion model for the source.

    source_tag = al.pipeline_settings.source_tag_from_source(
        source=af.last.instance.galaxies.source
    )

    pipeline_name = "pipeline_mass__power_law__lens_power_law__source_" + source_tag

    # This pipeline's name is tagged according to whether:

    # 1) Hyper-fitting settings (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag + pipeline_mass_settings.tag)

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and source, where we:

    # 1) Use the source galaxy of the 'source' pipeline.
    # 2) Set priors on the lens galaxy mass using the EllipticalIsothermal and ExternalShear of previous pipelines.

    # Setup the power-law mass profile and initialize its priors from the SIE.

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = af.last.model.galaxies.lens.mass.centre
    mass.axis_ratio = af.last.model.galaxies.lens.mass.axis_ratio
    mass.phi = af.last.model.galaxies.lens.mass.phi
    mass.einstein_radius = af.last.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

    # Setup the source model, which uses a variable parametric profile or fixed inversion model depending on the
    # previous pipeline.

    source = al.pipeline_settings.source_from_result(result=af.last)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_power_law__source_" + source_tag,
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=mass,
                shear=af.last.model.galaxies.lens.shear,
            ),
            source=source,
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    # If the source is parametric, the inversion hyper phase below will be skipped.

    phase1 = phase1.extend_with_multiple_hyper_phases(
        inversion=True,
        hyper_galaxy=pipeline_general_settings.hyper_galaxies,
        include_background_sky=pipeline_general_settings.hyper_image_sky,
        include_background_noise=pipeline_general_settings.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1)
