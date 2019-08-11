import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this pipeline, we'll perform a subhalo analysis which determines the sensitivity map of a strong lens and
# then attempts to detection subhalos by putting subhalos at fixed intevals on a 2D (y,x) grid. The source uses a
# light profile. The pipeline is as follows:

# Phase 1:

# Description: Perform the subhalo detection analysis.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: Lens mass (constant -> previous pipeline), source light (variable -> previous pipeline).
# Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 2:

# Description: Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Subhalo: SphericalTruncatedNFWChallenge
# Previous Pipelines: initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: Lens mass (variable -> previous pipeline), source light and subhalo mass (variable -> phase 2).
# Notes: None


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
    interp_pixel_scale=None,
    use_inversion_border=True,
    inversion_pixel_limit=None,
    cluster_pixel_scale=0.1,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_subhalo_hyper__lens_sie__subhalo_nfw__source_inversion"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
        hyper_galaxies=pipeline_settings.hyper_galaxies,
        hyper_image_sky=pipeline_settings.hyper_image_sky,
        hyper_background_noise=pipeline_settings.hyper_background_noise,
        include_shear=pipeline_settings.include_shear,
        pixelization=pipeline_settings.pixelization,
        regularization=pipeline_settings.regularization,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### Phase 1 ###

    # In phase 1, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model and source-pixelization parameters are held fixed to the best-fit values from phase 1 of the
    #    initialization pipeline.
    # 2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(af.as_grid_search(phase_imaging.PhaseImaging)):
        @property
        def grid_priors(self):
            return [
                self.variable.galaxies.subhalo.mass.centre_0,
                self.variable.galaxies.subhalo.mass.centre_1,
            ]

        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).constant.galaxies.mass

            self.galaxies.lens.shear = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).constant.galaxies.shear

            ### Lens Subhalo, Adjust priors to physical masses (10^6 - 10^10) and concentrations (6-24)

            self.galaxies.subhalo.mass.kappa_s = af.UniformPrior(
                lower_limit=0.0005, upper_limit=0.2
            )
            self.galaxies.subhalo.mass.scale_radius = af.UniformPrior(
                lower_limit=0.001, upper_limit=1.0
            )
            self.galaxies.subhalo.mass.centre_0 = af.UniformPrior(
                lower_limit=-2.0, upper_limit=2.0
            )
            self.galaxies.subhalo.mass.centre_1 = af.UniformPrior(
                lower_limit=-2.0, upper_limit=2.0
            )

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source.pixelization = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).hyper_combined.constant.galaxies.source.pixelization

            self.galaxies.source.regularization = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).hyper_combined.constant.galaxies.source.regularization

            ## Set all hyper_galaxy-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxies:

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

    phase1 = GridPhase(
        phase_name="phase_1__subhalo_search",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            subhalo=gm.GalaxyModel(
                redshift=redshift_lens, mass=mp.SphericalTruncatedNFWChallenge
            ),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pipeline_settings.pixelization,
                regularization=pipeline_settings.regularization,
            ),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        use_inversion_border=use_inversion_border,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
        number_of_steps=4,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.5

    class SubhaloPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ### Lens Mass, PL -> PL, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).variable.galaxies.lens

            ### Subhalo, TruncatedNFW -> TruncatedNFW ###

            self.galaxies.subhalo.mass.centre = (
                results.from_phase("phase_1__subhalo_search")
                .best_result.variable_absolute(a=0.3)
                .galaxies.subhalo.mass.centre
            )

            self.galaxies.subhalo.mass.kappa_s = (
                results.from_phase("phase_1__subhalo_search")
                .best_result.variable_relative(r=0.5)
                .galaxies.subhalo.mass.kappa_s
            )

            self.galaxies.subhalo.mass.scale_radius = (
                results.from_phase("phase_1__subhalo_search")
                .best_result.variable_relative(r=0.5)
                .galaxies.subhalo.mass.scale_radius
            )

            ### Source Light, Inversion -> Inversion ###

            self.galaxies.source.pixelization = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).hyper_combined.constant.galaxies.source.pixelization

            self.galaxies.source.regularization = results.from_phase(
                "phase_4__lens_sie__source_inversion"
            ).hyper_combined.constant.galaxies.source.regularization

            if pipeline_settings.hyper_galaxies:

                self.galaxies.source.hyper_galaxy = results.from_phase(
                    "phase_4__lens_sie__source_inversion"
                ).hyper_combined.constant.galaxies.source.hyper_galaxy

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = results.from_phase(
                    "phase_4__lens_sie__source_inversion"
                ).hyper_combined.constant.hyper_image_sky

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = results.from_phase(
                    "phase_4__lens_sie__source_inversion"
                ).hyper_combined.constant.hyper_background_noise

    phase2 = SubhaloPhase(
        phase_name="phase_2__subhalo_refine",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear,
            ),
            subhalo=gm.GalaxyModel(
                redshift=redshift_lens, mass=mp.SphericalTruncatedNFWChallenge
            ),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pipeline_settings.pixelization,
                regularization=pipeline_settings.regularization,
            ),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        use_inversion_border=use_inversion_border,
        inversion_pixel_limit=inversion_pixel_limit,
        cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 80
    phase2.optimizer.sampling_efficiency = 0.3

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=True,
    )

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, hyper_mode=True)
