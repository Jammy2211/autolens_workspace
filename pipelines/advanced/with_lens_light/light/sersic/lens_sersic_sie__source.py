import autofit as af
import autolens as al

# In this pipeline, we fit the lens light of a strong lens using a single Sersic model.

# The mass model and source are initialized using an already run 'source' pipeline. Although the lens light was
# fitted in this pipeline, we do not use this model to set priors in this pipeline.

# The pipeline is one phase:

# Phase 1:

# Fit the lens light using a Sersic model, with the lens mass and source fixed to the result of the previous pipeline.

# Lens Light & Mass: EllipticalSersic
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: Previous 'source' pipeline.
# Previous Pipelines: with_lens_light/source/*/lens_bulge_disk_sie__source_*.py
# Prior Passing: Lens Mass (instance -> previous pipeline), Source (instance -> previous pipeliine).
# Notes: Can be customized to vary the lens mass and source.


def source_is_inversion_from_setup(setup):
    if setup.source.type_tag in "sersic":
        return False
    else:
        return True


def source_with_previous_model_or_instance(setup):
    """Setup the source source model using the previous pipeline or phase results.

    This function is required because the source light model is not specified by the pipeline itself (e.g. the previous
    pipelines determines if the source was modeled using parametric light profiles or an inversion.

    If the source was parametric this function returns the source as a model, given that a parametric source should be
    fitted for simultaneously with the mass model.

    If the source was an inversion then it is returned as an instance, given that the inversion parameters do not need
    to be fitted for alongside the mass model.

    The bool include_hyper_source determines if the hyper-galaxy used to scale the sources noises is included in the
    model fitting.
    """

    if setup.source.type_tag in "sersic":

        return al.GalaxyModel(
            redshift=af.last.instance.galaxies.source.redshift,
            sersic=af.last.instance.galaxies.source.sersic,
            hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
        )

    else:

        return al.GalaxyModel(
            redshift=af.last.instance.galaxies.source.redshift,
            pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
            regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
            hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
        )


def make_pipeline(
    setup,
    phase_folders=None,
    redshift_lens=0.5,
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

    pipeline_name = "pipeline_light__sersic"

    # For pipeline tagging we need to set the light type
    setup.set_light_type(light_type="sersic")

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.general.tag)
    phase_folders.append(setup.source.tag)
    phase_folders.append(setup.light.tag)

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's light, where we:

    # 1) Fix the lens galaxy's mass and source galaxy to the results of the previous pipeline.
    # 2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.

    # If hyper-galaxy noise scaling is on, it may over-scale the noise making this new light profile fit the data less
    # well. This can be circumvented by including the noise scaling as a free parameter.

    if setup.general.hyper_galaxies:

        hyper_galaxy = af.PriorModel(al.HyperGalaxy)

        hyper_galaxy.noise_factor = (
            af.last.hyper_combined.model.galaxies.lens.hyper_galaxy.noise_factor
        )
        hyper_galaxy.contribution_factor = (
            af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = (
            af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.noise_power
        )

    else:

        hyper_galaxy = None

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lp.EllipticalSersic,
        mass=af.last.instance.galaxies.lens.mass,
        shear=af.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    source = source_with_previous_model_or_instance(setup=setup)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic_sie__source",
        phase_folders=phase_folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_pixel_limit=inversion_pixel_limit,
        inversion_uses_border=inversion_uses_border,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = False
    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.5
    phase1.optimizer.evidence_tolerance = 0.8

    # If the source is parametric, the inversion hyper phase below will be skipped.

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        inversion=source_is_inversion_from_setup(setup=setup),
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1)
