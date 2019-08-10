import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

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
    sub_grid_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    positions_threshold=None,
    inner_mask_radii=None,
    interp_pixel_scale=None,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_power_law__lens_bulge_disk_power_law__source_sersic"
    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
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

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ### Lens Light, Sersic -> Sersic, Exp -> Exp ###

            self.galaxies.lens.bulge = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.bulge

            self.galaxies.lens.disk = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.disk

            ### Lens Mass, SIE -> PL ###

            self.galaxies.lens.mass.centre = (
                results.from_phase("phase_4__lens_bulge_disk_sie__source_sersic")
                .variable_absolute(a=0.05)
                .galaxies.lens.mass.centre
            )

            self.galaxies.lens.mass.axis_ratio = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.mass.axis_ratio

            self.galaxies.lens.mass.phi = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.lens.mass.phi

            self.galaxies.lens.mass.einstein_radius = (
                results.from_phase("phase_4__lens_bulge_disk_sie__source_sersic")
                .variable_absolute(a=0.3)
                .galaxies.lens.mass.einstein_radius
            )

            ### Lens Shear, Shear -> Shear ###

            if pipeline_settings.include_shear:

                self.galaxies.lens.shear = results.from_phase(
                    "phase_4__lens_bulge_disk_sie__source_sersic"
                ).variable.galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_sersic"
            ).variable.galaxies.source

    phase1 = LensSourcePhase(
        phase_name="phase_1__lens_bulge_disk_power_law__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge=lp.EllipticalSersic,
                disk=lp.EllipticalExponential,
                mass=mp.EllipticalPowerLaw,
                shear=mp.ExternalShear,
            ),
            source=gm.GalaxyModel(redshift=redshift_source, light=lp.EllipticalSersic),
        ),
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    return pipeline.PipelineImaging(pipeline_name, phase1)
