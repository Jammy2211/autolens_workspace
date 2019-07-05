import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging, phase_extensions
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import mass_profiles as mp
from autolens.model.profiles import light_and_mass_profiles as lmp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os


# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using an inversion, using a decomposed light and dark matter profile. The pipeline follows on from the
# inversion pipeline 'pipelines/with_lens_light/inversion/from_initialize/lens_sersic_exp_sie_shear_source_inversion.py'.

# Alignment of the centre, phi and axis-ratio of the light profile's EllipticalSersic and EllipticalExponential
# profiles use the alignment specified in the previous pipeline.

# The pipeline is two phases, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a decomposed profile, using an inversion for the Source.
# Lens Light & Mass: EllipticalSersic + EllipticalExponential
# Lens Mass: SphericalNFW + ExternalShear
# Source Light: VoronoiMagnification + Constant
# Previous Pipelines: with_lens_light/bulge_disk/inversion/from_initialize/lens_sersic_exp_sieexp_shear_source_inversion.py
# Prior Passing: Lens Light (variable -> previous pipeline), Lens Mass (default),
#                Source Inversion (variable / constant -> previous pipeline)
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.

def make_pipeline(
        pl_align_bulge_dark_centre=True,
        pl_hyper_galaxies=True, pl_hyper_background_sky=True,  pl_hyper_background_noise=True,
        pl_pixelization=pix.VoronoiBrightnessImage, pl_regularization=reg.AdaptiveBrightness,
        phase_folders=None, tag_phases=True,
        redshift_lens=0.5, redshift_source=1.0,
        sub_grid_size=2, bin_up_factor=None, positions_threshold=None, inner_mask_radii=None, interp_pixel_scale=0.05,
        inversion_pixel_limit=None, cluster_pixel_scale=0.1):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pipeline_ldm__lens_sersic_exp_mlr_nfw_shear_source_inversion'

    pipeline_name = tag.pipeline_name_from_name_and_settings(
        pipeline_name=pipeline_name, pixelization=pl_pixelization, regularization=pl_regularization)

    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Pass priors on the lens galaxy's light using the EllipticalSersic and EllipticalExponential of the previous 
    #    pipeline. This includes using the bulge-disk alignment assumed in that pipeline.
    # 2) Pass priors on the lens galaxy's SphericalNFW mass profile's centre using the EllipticalIsothermal fit of the
    #    previous pipeline, if the NFW centre is a free parameter.
    # 3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
    # 4) Pass priors on the source galaxy's inversion using the Pixelization and Regularization of the previous pipeline.

    class LensSourcePhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Light, Sersic -> Sersic, Exp -> Exp ###
            
            self.lens_galaxies.lens.bulge = results.from_phase('phase_2_lens_sersic_exp_sie_shear_source_inversion').\
                variable.lens_galaxies.lens.bulge

            self.lens_galaxies.lens.disk = results.from_phase('phase_2_lens_sersic_exp_sie_shear_source_inversion').\
                variable.lens_galaxies.lens.disk

            ### Lens Mass, SIE ->  NFW ###

            if pl_align_bulge_dark_centre:

                self.lens_galaxies.lens.dark.centre = self.lens_galaxies.bulge.centre

            elif not pl_align_bulge_dark_centre:

                self.lens_galaxies.lens.dark.centre = results.from_phase('phase_2_lens_sersic_exp_sie_shear_source_inversion').\
                    variable_absolute(a=0.05).lens_galaxies.lens.mass.centre

            ### Lens Shear, Shear -> Shear ###

            self.lens_galaxies.lens.shear = results.from_phase('phase_2_lens_sersic_exp_sie_shear_source_inversion').\
                variable.lens_galaxies.lens.shear

            ### Source Inversion, Inv -> Inv ###

            self.source_galaxies.source = results.from_phase('phase_2_lens_sersic_exp_sie_shear_source_inversion').inversion.\
                constant.source_galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            if pl_hyper_galaxies:

                self.lens_galaxies.lens.hyper_galaxy = results.last.hyper_galaxy. \
                    constant.lens_galaxies.lens.hyper_galaxy

                self.source_galaxies.source.hyper_galaxy = results.last.hyper_galaxy. \
                    constant.source_galaxies.source.hyper_galaxy

            if pl_hyper_background_sky:

                self.hyper_image_sky = results.last.inversion. \
                    constant.hyper_image_sky

            if pl_hyper_background_noise:

                self.hyper_noise_background = results.last.inversion. \
                    constant.hyper_noise_background

    phase1 = LensSourcePhase(
        phase_name='phase_1_lens_sersic_exp_mlr_nfw_shear_source_inversion', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge=lmp.EllipticalSersic,
                disk=lmp.EllipticalExponential,
                mass=mp.EllipticalPowerLaw,
                shear=mp.ExternalShear)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pl_pixelization,
                regularization=pl_regularization)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, interp_pixel_scale=interp_pixel_scale,
        inversion_pixel_limit=inversion_pixel_limit, cluster_pixel_scale=cluster_pixel_scale,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_hyper_and_inversion_phases(
        hyper_galaxy=pl_hyper_galaxies,
        include_background_sky=pl_hyper_background_sky,
        include_background_noise=pl_hyper_background_noise,
        inversion=True)

    return pipeline.PipelineImaging(pipeline_name, phase1)