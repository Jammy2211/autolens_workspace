import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import mass_profiles as mp
from autolens.model.profiles import light_and_mass_profiles as lmp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os


# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using an inversion, using a decomposed light and dark matter profile. The pipeline follows on from the
# inversion pipeline 'pipelines/with_lens_light/inversion/from_initialize/lens_bulge_disk_sie__source_inversion.py'.

# Alignment of the centre, phi and axis-ratio of the light profile's EllipticalSersic and EllipticalExponential
# profiles use the alignment specified in the previous pipeline.

# The pipeline is two phases, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a decomposed profile, using an inversion for the Source.
# Lens Light & Mass: EllipticalSersic + EllipticalExponential
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: with_lens_light/bulge_disk/inversion/from_initialize/lens_bulge_disk_sieexp_source_inversion.py
# Prior Passing: Lens Light (variable -> previous pipeline), Lens Mass (default),
#                Source Inversion (variable / constant -> previous pipeline)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.

# Phase 2:

# Description: Refines the inversion parameters, using a fixed mass model from phase 1.
# Lens Light & Mass: EllipticalSersic
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: None
# Prior Passing: Lens Light & Mass (constant -> phase 1), source inversion (variable -> phase 1)
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
    interp_pixel_scale=0.05,
    use_inversion_border=True,
    inversion_pixel_limit=None,
    cluster_pixel_scale=0.1,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_ldm__lens_bulge_disk_mlr_nfw__source_inversion"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
        include_shear=pipeline_settings.include_shear,
        align_bulge_disk_centre=pipeline_settings.align_bulge_disk_centre,
        align_bulge_disk_phi=pipeline_settings.align_bulge_disk_phi,
        align_bulge_disk_axis_ratio=pipeline_settings.align_bulge_disk_axis_ratio,
        disk_as_sersic=pipeline_settings.disk_as_sersic,
        align_bulge_dark_centre=pipeline_settings.align_bulge_dark_centre,
        pixelization=pipeline_settings.pixelization,
        regularization=pipeline_settings.regularization,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Pass priors on the lens galaxy's light using the EllipticalSersic and EllipticalExponential of the previous
    #    pipeline. This includes using the bulge-disk alignment assumed in that pipeline.
    # 2) Pass priors on the lens galaxy's SphericalNFW mass profile's centre using the EllipticalIsothermal fit of the
    #    previous pipeline, if the NFW centre is a free parameter.
    # 3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
    # 4) Pass priors on the source galaxy's inversion using the Pixelization and Regularization of the previous pipeline.

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ### Lens Light to Light + Mass, Sersic -> Sersic ###

            self.galaxies.lens.bulge_mass.centre = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.bulge.centre

            self.galaxies.lens.bulge_mass.axis_ratio = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.bulge.axis_ratio

            self.galaxies.lens.bulge_mass.phi = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.bulge.phi

            self.galaxies.lens.bulge_mass.intensity = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.bulge.intensity

            self.galaxies.lens.bulge_mass.effective_radius = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.bulge.effective_radius

            self.galaxies.lens.bulge_mass.sersic_index = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.bulge.sersic_index

            ### Lens Light to Light + Mass, Sersic -> Sersic, Exp -> Exp ###

            self.galaxies.lens.disk_mass.centre = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.disk.centre

            self.galaxies.lens.disk_mass.axis_ratio = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.disk.axis_ratio

            self.galaxies.lens.disk_mass.phi = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.disk.phi

            self.galaxies.lens.disk_mass.intensity = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.disk.intensity

            self.galaxies.lens.disk_mass.effective_radius = results.from_phase(
                "phase_4__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.disk.effective_radius

            ### Lens Mass, SIE ->  NFW ###

            if pipeline_settings.align_bulge_dark_centre:

                self.galaxies.lens.dark.centre = self.galaxies.lens.bulge_mass.centre

            elif not pipeline_settings.align_bulge_dark_centre:

                self.galaxies.lens.dark.centre = (
                    results.from_phase("phase_4__lens_bulge_disk_sie__source_inversion")
                    .variable_absolute(a=0.05)
                    .galaxies.lens.bulge_mass.centre
                )

            ### Lens Shear, Shear -> Shear ###

            self.galaxies.lens.shear = results.from_phase(
                "phase_2__lens_bulge_disk_sie__source_inversion"
            ).variable.galaxies.lens.shear

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_2__lens_bulge_disk_sie__source_inversion"
            ).inversion.constant.galaxies.source

    phase1 = LensSourcePhase(
        phase_name="phase_1__lens_bulge_disk_mlr_nfw__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge_mass=lmp.EllipticalSersic,
                disk_mass=lmp.EllipticalExponential,
                mass=mp.EllipticalPowerLaw,
                shear=mp.ExternalShear,
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

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_inversion_phase()

    return pipeline.PipelineImaging(pipeline_name, phase1)
