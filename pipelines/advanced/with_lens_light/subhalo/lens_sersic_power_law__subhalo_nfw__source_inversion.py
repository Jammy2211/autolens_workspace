import autofit as af
import autolens as al

# In this pipeline, we'll perform a subhalo analysis which determines the sensitivity map of a strong lens and
# then attempts to detection subhalos by putting subhalos at fixed intevals on a 2D (y,x) grid. The source uses an
# inversion. The pipeline is as follows:

# Phase 1:

# Description: Perform the subhalo detection analysis.
# Lens Light: EllipticalSersic
# Lens Mass: EllipticalPowerLaw + ExternalShear
# Source Light: VoronoiBrightnessImage + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: source/parametric/lens_sie__source_inversion_from_pipeline.py
# Prior Passing: Lens light, mass (instance -> previous pipeline), source inversion (model -> previous pipeline).
# Notes: Uses the lens subtracted image of a previous pipeline.
#        Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 2:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Light: EllipticalSersic
# Lens Mass: EllipticalPowerLaw + ExternalShear
# Source Light: VoronoiBrightnessImage + Constant
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: source/parametric/lens_sie__source_inversion_from_pipeline.py
# Prior Passing: Lens light and mass (model -> previous pipeline), source inversion and subhalo mass (model -> phase 2).
# Notes: None


def make_pipeline(
    pipeline_general_settings,
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

    pipeline_name = (
        "pipeline_subhalo_hyper__lens_sersic_sie__subhalo_nfw__source_inversion_"
        + source_tag
    )

    # 4) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_general_settings.tag)

    ### Phase 1 ###

    # In phase 1, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model parameters are held fixed to the best-fit values from phase 1 of the initialization pipeline.
    # 2) The source-pixelization resolution is fixed to the best-fit values the inversion source__parametricd pipeline, the
    #    regularized coefficienct is free to vary.
    # 3) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 4) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(af.as_grid_search(al.PhaseImaging)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(
        redshift=redshift_lens, mass=al.mp.SphericalTruncatedNFWMassToConcentration
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)

    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    phase1 = GridPhase(
        phase_name="phase_1__subhalo_search",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=af.last.instance.galaxies.lens.light,
                mass=af.last.instance.galaxies.lens.mass,
                shear=af.last.instance.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            subhalo=subhalo,
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=af.last.instance.galaxies.source.pixelization,
                regularization=af.last.instance.galaxies.source.regularization,
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
        number_of_steps=4,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 20
    phase1.optimizer.sampling_efficiency = 0.3

    phase2 = al.PhaseImaging(
        phase_name="phase_2__subhalo_refine",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=af.last[-1].model.galaxies.lens.light,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            subhalo=al.GalaxyModel(
                redshift=redshift_lens, mass=phase1.result.model.galaxies.subhalo.mass
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
    phase2.optimizer.n_live_points = 80
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        inversion=True,
        hyper_galaxy=pipeline_general_settings.hyper_galaxies,
        include_background_sky=pipeline_general_settings.hyper_image_sky,
        include_background_noise=pipeline_general_settings.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
