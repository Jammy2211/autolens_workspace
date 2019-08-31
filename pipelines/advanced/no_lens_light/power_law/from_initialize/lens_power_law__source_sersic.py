import autofit as af
import autolens as al

# In this pipeline, we'll perform an analysis which fits an image with  no lens light, and a source galaxy using a
# parametric light profile, using a power-law mass profile. The pipeline follows on from the initialize pipeline
# ''pipelines/no_lens_light/initialize/lens_sie__source_sersic_from_init.py'.

# The pipeline is one phase, as follows:

# Phase 1:

# Description: Fits the lens mass model as a power-law, using a parametric Sersic light profile for the source.
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: no_lens_light/initialize/lens_sie__source_sersic_from_init.py
# Prior Passing: None
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations.


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
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_power_law__lens_power_law__source_sersic"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        include_shear=pipeline_settings.include_shear
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy mass using the EllipticalIsothermal fit of the previous pipeline, and
    #    source galaxy of the previous pipeline.

    class LensSourcePhase(al.PhaseImaging):
        def customize_priors(self, results):

            ### Lens Mass, SIE -> PL ###

            self.galaxies.lens.mass.centre = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass.centre

            self.galaxies.lens.mass.axis_ratio = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass.axis_ratio

            self.galaxies.lens.mass.phi = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass.phi

            self.galaxies.lens.mass.einstein_radius = (
                results.from_phase("phase_1__lens_sie__source_sersic")
                .variable_absolute(a=0.3)
                .galaxies.lens.mass.einstein_radius
            )

            ### Lens Shear, Shear -> Shear ###

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_1__lens_sie__source_sersic"
                ).variable.galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.source

    phase1 = LensSourcePhase(
        phase_name="phase_1__lens_power_law__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=al.mass_profiles.EllipticalPowerLaw,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, light=al.light_profiles.EllipticalSersic
            ),
        ),
        sub_grid_size=sub_grid_size,
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

    return al.PipelineImaging(pipeline_name, phase1)
