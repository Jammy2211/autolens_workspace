import autofit as af
import autolens as al

# In this pipeline, we'll perform a source inversion analysis which fits an image with a source galaxy and a lens light
# component.

# The lens light model used depends on the previous 'source/parametric' pipeline that is run. For example, if the
# bulge-disk pipeline was used the bulge-disk model will be used in this pipeline.

# Phases 1 & 2 first use a magnification based pixelization and constant regularization scheme to reconstruct the
# source (as opposed to immediately using the pixelization & regularization input via the pipeline setup).
# This ensures that if the input pixelization or regularization scheme uses hyper-images, they are initialized using
# a pixelized source-plane, which is key for lens's with multiple or irregular sources.

# The pipeline is as follows:

# Phase 1:

# Set inversion's pixelization and regularization, using a magnification
# based pixel-grid and the previous lens light and mass model.

# Lens Light: Previous Pipeline.
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: source/parametric/lens_bulge_disk_sie__source_sersic.py
# Prior Passing: Lens Light / Mass (instance -> previous pipeline).
# Notes: Lens light & mass fixed, source inversion parameters vary.

# Phase 2:

# Refine the lens mass model using the source inversion.

# Lens Light: Previous Pipeline.
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: source/parametric/lens_bulge_disk_sie__source_sersic.py
# Prior Passing: Lens Light & Mass (model -> previous pipeline), source inversion (instance -> phase 1).
# Notes: Lens light fixed, mass varies, source inversion parameters fixed.

# Phase 3:

# Fit the inversion's pixelization and regularization, using the input pixelization,
# regularization and the previous lens mass model.

# Lens Light: Previous Pipeline.
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: setup.pixelization + setup.regularization
# Previous Pipelines: None
# Prior Passing: Lens Light & Mass (instance -> phase 2).
# Notes:  Lens light & mass fixed, source inversion parameters vary.

# Phase 4:

# Refine the lens mass model using the inversion.

# Lens Light: Previous Pipeline.
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: pixelization + regularization
# Prior Passing: Lens Light & Mass (model -> phase 3), source inversion (instance -> phase 3).
# Notes: Lens light fixed, mass varies, source inversion parameters fixed.


def lens_light_tag_from_setup(setup):

    if setup.light.type_tag is "":
        return ""
    else:
        return "__light_" + setup.light.type_tag


def lens_with_previous_light_and_model_mass(setup):
    """Setup the lens galaxy model using the previous pipeline or phases results.

    This function is required because the lens light model is not specified by the pipeline itself (e.g. this previous
    pipeline determines if the lens light was modeled as a Sersic, bulge-disk, Gaussians, etc.)

    So, we have to pass the lens light to this pipeline without explicitly referencing its light components."""

    if setup.source.fix_lens_light:
        lens = af.last[-1].instance.galaxies.lens
        lens.mass = af.last[-1].model.galaxies.lens.mass
        lens.shear = af.last[-1].model.galaxies.lens.shear
    else:
        lens = af.last[-1].model.galaxies.lens

    lens.hyper_galaxy = (
        af.last.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy
    )

    return lens


def make_pipeline(
    setup,
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
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline_source__inversion"

    # For pipeline tagging we need to set the source type
    setup.set_source_type(source_type=setup.source.inversion_tag)

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The pixelization and regularization scheme of the pipeline (fitted in phases 3 & 4).
    # 3) The lens light model is fixed during the analysis.
    # 4) The lens galaxy mass model includes an external shear.
    # 5) The lens light model used in the previous pipeline.

    lens_light_tag_from_setup(setup=setup)

    phase_folders.append(pipeline_name)
    if setup.light.type_tag is not "":
        light_tag = "__" + setup.light.type_tag
    else:
        light_tag = setup.light.type_tag

    phase_folders.append(setup.general.tag + light_tag)
    phase_folders.append(setup.source.tag)

    ### PHASE 1 ###

    # In phase 1, we fit the pixelization and regularization, where we:

    # 1) Fix the lens light & mass model to the light & mass models inferred by the previous pipeline.

    # This function is used through the pipeline - it setups the lens light model using the light and mass profiles
    # assumed in the previous pipeline. They are setup as models or instances depending on the 'fix_lens_'light' and
    # 'fix_lens_mass' boolean inputs.

    #  lens = af.last.instance.galaxies.lens
    #  lens.hyper_galaxy = af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy

    phase1 = al.PhaseImaging(
        phase_name="phase_1__source_inversion_magnification_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last.instance.galaxies.lens.bulge,
                disk=af.last.instance.galaxies.lens.disk,
                mass=af.last.instance.galaxies.lens.mass,
                shear=af.last.instance.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
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
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 2 ###

    # In phase 2, we refine the len galaxy mass using an inversion. We will:

    # 1) Fix the source inversion parameters to the results of phase 1.
    # 2) Fix the lens light model to the results of the previous pipeline.
    # 3) Set priors on the lens galaxy mass from the previous pipeline.

    # lens = lens_with_previous_light_and_model_mass(setup=setup)

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_light_sie__source_inversion_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=af.last[-1].instance.galaxies.lens.bulge,
                disk=af.last[-1].instance.galaxies.lens.disk,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
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
    phase2.optimizer.evidence_tolerance = evidence_tolerance

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 3 ###

    # In phase 3, we fit the input pipeline pixelization & regularization, where we:

    # 1) Fix the lens light model to the results of the previous pipeline.
    # 2) Fix the lens mass model to the mass-model inferred in phase 2.

    # lens = phase2.result.instance.galaxies.lens
    # lens.hyper_galaxy = (
    #     phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy
    # )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__source_inversion_initialization",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase2.result.instance.galaxies.lens.bulge,
                disk=phase2.result.instance.galaxies.lens.disk,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=setup.source.pixelization,
                regularization=setup.source.regularization,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 20
    phase3.optimizer.sampling_efficiency = 0.8

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 4 ###

    # In phase 4, we fit the lens's mass using the input pipeline pixelization & regularization, where we:

    # 1) Fix the source inversion parameters to the results of phase 3.
    # 2) Fix the lens light model to the results of the previous pipeline.
    # 3) Set priors on the lens galaxy mass using the results of phase 2.

    # lens = lens_with_previous_light_and_model_mass(setup=setup)

    # If they were aligned, unalign the lens light and mass given the model is now initialized.

    # TODO : Generaalize this if loop to any lens light profile

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    if setup.source.lens_mass_centre is not None:

        mass.centre.centre_0 = af.GaussianPrior(
            mean=setup.source.lens_mass_centre[0], sigma=0.05
        )
        mass.centre.centre_1 = af.GaussianPrior(
            mean=setup.source.lens_mass_centre[1], sigma=0.05
        )

    elif setup.source.align_light_mass_centre:

        mass.centre = af.last[-3].model.galaxies.lens.bulge.centre

    mass.axis_ratio = phase2.result.model.galaxies.lens.mass.axis_ratio
    mass.phi = phase2.result.model.galaxies.lens.mass.phi
    mass.einstein_radius = phase2.result.model.galaxies.lens.mass.einstein_radius

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=phase2.result.instance.galaxies.lens.bulge,
        disk=phase2.result.instance.galaxies.lens.disk,
        mass=mass,
        shear=phase2.result.model.galaxies.lens.shear,
        hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_light_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase3.result.instance.galaxies.source.pixelization,
                regularization=phase3.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 75
    phase4.optimizer.sampling_efficiency = 0.2
    phase4.optimizer.evidence_tolerance = evidence_tolerance

    phase4 = phase4.extend_with_multiple_hyper_phases(
        inversion=True,
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
