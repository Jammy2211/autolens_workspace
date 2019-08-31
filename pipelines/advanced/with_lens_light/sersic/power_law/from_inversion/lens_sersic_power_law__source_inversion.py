import autofit as af
import autolens as al

# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using an inversion, using a power-law mass profile. The pipeline follows on from the inversion pipeline
# ''pipelines/with_lens_light/inversion/from_initialize/lens_sersic_sie__source_inversion.py'.

# The pipeline is two phases, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a power-law, using an inversion for the Source.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: with_lens_light/inversion/from_initialize/lens_sersic_sie__source_inversion.py
# Prior Passing: Lens Light (variable -> previous pipeline), Lens Mass (variable -> previous pipeline),
#                Source Inversion (variable / constant -> previous pipeline)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.

# Phase 2:

# Description: Refines the inversion parameters, using a fixed mass model from phase 1.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: None
# Prior Passing: Lens Mass (constant -> phase 1), source inversion (variable -> phase 1)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.


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
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    pixel_scale_binned_cluster_grid=0.1,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_power_law__lens_sersic_power_law__source_inversion"
    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        include_shear=pipeline_settings.include_shear,
        fix_lens_light=pipeline_settings.fix_lens_light,
        pixelization=pipeline_settings.pixelization,
        regularization=pipeline_settings.regularization,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Pass priors on the lens galaxy's light using the EllipticalSersic of the previous pipeline.
    # 2) Pass priors on the lens galaxy's  mass using the EllipticalIsothermal and ExternalShear fit of the previous
    #    pipeline.
    # 3) Pass priors on the source galaxy's inversion using the Pixelization and Regularization of the previous pipeline.

    class LensSourcePhase(al.PhaseImaging):
        def customize_priors(self, results):

            ### Lens Light, Sersic -> Sersic ###

            self.galaxies.lens.light = results.from_phase(
                "phase_2__lens_sersic_sie__source_inversion"
            ).variable.galaxies.light

            ### Lens Mass, SIE -> Powerlaw ###

            self.galaxies.lens.mass.centre = (
                results.from_phase("phase_2__lens_sersic_sie__source_inversion")
                .variable_absolute(a=0.05)
                .galaxies.lens.mass.centre
            )

            self.galaxies.lens.mass.axis_ratio = results.from_phase(
                "phase_2__lens_sersic_sie__source_inversion"
            ).variable.galaxies.lens.mass.axis_ratio

            self.galaxies.lens.mass.phi = results.from_phase(
                "phase_2__lens_sersic_sie__source_inversion"
            ).variable.galaxies.lens.mass.phi

            self.galaxies.lens.mass.einstein_radius = (
                results.from_phase("phase_2__lens_sersic_sie__source_inversion")
                .variable_absolute(a=0.3)
                .galaxies.lens.mass.einstein_radius
            )

            ### Lens Shear, Shear -> Shear ###

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_2__lens_sersic_sie__source_inversion"
                ).variable.galaxies.lens.shear

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_2__lens_sersic_sie__source_inversion"
            ).inversion.constant.galaxies.source

    phase1 = LensSourcePhase(
        phase_name="phase_1__lens_sersic_power_law__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalPowerLaw,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
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
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        pixel_scale_binned_cluster_grid=pixel_scale_binned_cluster_grid,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_inversion_phase()

    return al.PipelineImaging(pipeline_name, phase1)
