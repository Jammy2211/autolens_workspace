import autofit as af
import autolens as al

# In this pipeline, we'll perform a parametric source analysis which fits a lens model (the lens's light, mass and
# source's light). The lens's light is modeled as a sum of elliptical Gaussian profiles. This pipeline uses four phases:

# Phase 1:

# Fit and subtract the lens light model.

# Lens Light: EllipticalGaussian(s)
# Lens Mass: None
# Source Light: None
# Previous Pipelines: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Fit the lens mass model and source light profile, using the lens subtracted image from phase 1.

# Lens Light: None
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses the lens light subtracted image from phase 1

# Phase 3:

# Refit the lens light models using the mass model and source light profile fixed from phase 2.

# Lens Light: EllticalGaussian(s)
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: lens mass and source (instance -> phase 2)
# Notes: None

# Phase 4:

# Refine the lens light and mass models and source light profile, using priors from the previous 2 phases.

# Lens Light: EllipticalGaussian(s)
# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens light (model -> phase 3), lens mass and source (model -> phase 2)
# Notes: None


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
    evidence_tolerance=100.0,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline_source__parametric__lens_gaussians"

    # For pipeline tagging we need to set the source type
    setup.set_source_type(source_type="sersic")
    setup.set_light_type(light_type="gaussians")

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.
    # 3) The number of Gaussians in the lens light model.

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.general.tag)
    phase_folders.append(setup.source.tag)

    ### SETUP SHEAR ###

    # Include the shear in the mass model if not switched off in the pipeline setup.

    if not setup.source.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # In phase 1, we fit only the lens galaxy's light, where we:

    # 1) Align the Gaussian's (y,x) centres.

    gaussian_0 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_1 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_2 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_3 = af.PriorModel(al.lp.EllipticalGaussian)

    gaussian_1.centre = gaussian_0.centre
    gaussian_2.centre = gaussian_0.centre
    gaussian_3.centre = gaussian_0.centre

    if setup.source.lens_light_centre is not None:
        gaussian_0.centre = setup.source.lens_light_centre
        gaussian_1.centre = setup.source.lens_light_centre
        gaussian_2.centre = setup.source.lens_light_centre
        gaussian_3.centre = setup.source.lens_light_centre

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        gaussian_0=gaussian_0,
        gaussian_1=gaussian_1,
        gaussian_2=gaussian_2,
        gaussian_3=gaussian_3,
    )

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_gaussians",
        phase_folders=phase_folders,
        galaxies=dict(lens=lens),
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.3
    phase1.optimizer.evidence_tolerance = evidence_tolerance

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and source galaxy's light, where we:

    # 1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
    # 2) Set priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for \
    #    the bulge of the light profile in phase 1.

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    if setup.source.align_light_mass_centre:
        mass.centre = phase1.result.instance.galaxies.lens.gaussian_0.centre
    else:
        mass.centre = phase1.result.model_absolute(
            a=0.1
        ).galaxies.lens.gaussian_0.centre

    if setup.source.lens_mass_centre is not None:
        mass.centre = setup.source.lens_mass_centre

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                gaussian_0=phase1.result.instance.galaxies.lens.gaussian_0,
                gaussian_1=phase1.result.instance.galaxies.lens.gaussian_1,
                gaussian_2=phase1.result.instance.galaxies.lens.gaussian_2,
                gaussian_3=phase1.result.instance.galaxies.lens.gaussian_3,
                mass=mass,
                shear=shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=al.lp.EllipticalSersic,
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
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.5
    phase2.optimizer.evidence_tolerance = evidence_tolerance

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    ### PHASE 3 ###

    # In phase 3, we will refit the lens galaxy's light using a fixed mass and source model above, where we:

    # 1) Do not use priors from phase 1 to Fit the lens's light, assuming the source light may of impacted them.

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        gaussian_0=gaussian_0,
        gaussian_1=gaussian_1,
        gaussian_2=gaussian_2,
        gaussian_3=gaussian_3,
        mass=phase2.result.instance.galaxies.lens.mass,
        shear=phase2.result.instance.galaxies.lens.shear,
        hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_gaussians_sie__source_fixed",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase2.result.instance.galaxies.source.sersic,
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
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = False
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.5
    phase3.optimizer.evidence_tolerance = evidence_tolerance

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    ### PHASE 4 ###

    # In phase 4, we fit simultaneously the lens and source galaxies, where we:

    # 1) Set lens's light, mass, shear and source's light using the results of phases 1 and 2.

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_gaussians_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                gaussian_0=phase3.result.model.galaxies.lens.gaussian_0,
                gaussian_1=phase3.result.model.galaxies.lens.gaussian_1,
                gaussian_2=phase3.result.model.galaxies.lens.gaussian_2,
                gaussian_3=phase3.result.model.galaxies.lens.gaussian_3,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                sersic=phase2.result.model.galaxies.source.sersic,
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
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 75
    phase4.optimizer.sampling_efficiency = 0.3
    phase4.optimizer.evidence_tolerance = evidence_tolerance

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
