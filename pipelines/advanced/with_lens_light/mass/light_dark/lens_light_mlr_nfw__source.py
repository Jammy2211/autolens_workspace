import autofit as af
import autolens as al

# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using an inversion, using a decomposed light and dark matter profile. The pipeline follows on from the
# inversion pipeline 'pipelines/with_lens_light/inversion/from_source__parametric/lens_light_sie__source_inversion.py'.

# Alignment of the centre, phi and axis-ratio of the light profile's EllipticalSersic and EllipticalExponential
# profiles use the alignment specified in the previous pipeline.

# The pipeline is two phases:

# Phase 1:

# Description: Fit the lens light and mass model as a decomposed profile, using an inversion for the Source.
# Lens Light & Mass: EllipticalSersic + EllipticalExponential
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: with_lens_light/bulge_disk/inversion/from_source__parametric/lens_bulge_disk_sieexp_source_inversion.py
# Prior Passing: Lens Light (model -> previous pipeline), Lens Mass (default),
#                Source Inversion (model / instance -> previous pipeline)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.

# Phase 2:

# Description: Refines the inversion parameters, using a fixed mass model from phase 1.
# Lens Light & Mass: EllipticalSersic
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: None
# Prior Passing: Lens Light & Mass (instance -> phase 1), source inversion (model -> phase 1)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.


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
    if setup.general.hyper_galaxies:

        hyper_galaxy = af.PriorModel(al.HyperGalaxy)

        hyper_galaxy.noise_factor = (
            af.last.hyper_combined.model.galaxies.source.hyper_galaxy.noise_factor
        )
        hyper_galaxy.contribution_factor = (
            af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = (
            af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.noise_power
        )

    else:

        hyper_galaxy = None

    if setup.source.type_tag in "sersic":

        return al.GalaxyModel(
            redshift=af.last.instance.galaxies.source.redshift,
            sersic=af.last.model.galaxies.source.sersic,
            hyper_galaxy=hyper_galaxy,
        )

    else:

        return al.GalaxyModel(
            redshift=af.last.instance.galaxies.source.redshift,
            pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
            regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
            hyper_galaxy=hyper_galaxy,
        )


def make_pipeline(
    setup,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    pixel_scale_interpolation_grid=0.05,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    # A source tag distinguishes if the previous pipeline models used a parametric or inversion model for the source.

    pipeline_name = "pipeline_mass__light_dark__bulge_disk"

    # For pipeline tagging we need to set the mass type
    setup.set_mass_type(mass_type="light_dark")

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The bulge + disk centres, rotational angles or axis ratios are aligned.
    # 3) The disk component of the lens light model is an Exponential or Sersic profile.
    # 4) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.general.tag)
    phase_folders.append(setup.source.tag)
    phase_folders.append(setup.light.tag)
    phase_folders.append(setup.mass.tag)

    ### SETUP SHEAR ###

    # Include the shear in the mass model if not switched off in the pipeline setup.

    if not setup.mass.no_shear:
        if af.last.model.galaxies.lens.shear is not None:
            shear = af.last.model.galaxies.lens.shear
        else:
            shear = al.mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Fix the lens galaxy's light using the the light profile inferred in the previous 'light' pipeline, including
    #    assumptions related to the geometric alignment of different components.
    # 2) Pass priors on the lens galaxy's SphericalNFW mass profile's centre using the EllipticalIsothermal fit of the
    #    previous pipeline, if the NFW centre is a free parameter.
    # 3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
    # 4) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.

    if setup.light.disk_as_sersic:
        disk = af.PriorModel(al.lmp.EllipticalSersic)
    else:
        disk = af.PriorModel(al.lmp.EllipticalExponential)

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lmp.EllipticalSersic,
        disk=disk,
        dark=al.mp.SphericalNFW,
        shear=shear,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
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
    if setup.light.disk_as_sersic:
        lens.disk.sersic_index = af.last.instance.galaxies.lens.disk.sersic_index

    lens.dark.scale_radius = af.GaussianPrior(mean=30.0, sigma=5.0)

    if setup.mass.align_bulge_dark_centre:
        lens.dark.centre = lens.bulge.centre
    else:
        lens.dark.centre = af.last.model_absolute(a=0.05).galaxies.lens.bulge.centre

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_light_mlr_nfw__source__fixed_lens_light",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
                regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = 0.8

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's light and mass and source galaxy using the results of phase 1 as
    # initialization

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_light_mlr_nfw__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase1.result.model.galaxies.lens.bulge,
                disk=phase1.result.model.galaxies.lens.disk,
                dark=phase1.result.model.galaxies.lens.dark,
                shear=phase1.result.model.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 75
    phase2.optimizer.sampling_efficiency = 0.2
    phase2.optimizer.evidence_tolerance = 0.8

    phase2 = phase2.extend_with_inversion_phase()

    return al.PipelineDataset(pipeline_name, phase1, phase2)
