import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging, phase_extensions
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

# In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
# using a parametric light profile, using a power-law mass profile. The pipeline follows on from the initialize pipeline
# ''pipelines/no_lens_light/initialize/lens_sersic_ie_shear_source_sersic_from_init.py'.

# The pipeline is one phase, as follows:

# Phase 1:

# Description: Fits the lens light and mass model as a power-law, using a parametric Sersic light profile for the source.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: with_lens_light/initialize/lens_sersic_sie_shear_source_sersic.py
# Prior Passing: None
# Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.

def make_pipeline(
        pl_hyper_galaxies=True,
        phase_folders=None, tag_phases=True,
        redshift_lens=0.5, redshift_source=1.0,
        sub_grid_size=2, bin_up_factor=None, positions_threshold=None, inner_mask_radii=None, interp_pixel_scale=0.05):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pipeline_pl__lens_sersic_pl_shear_source_sersic'

    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's light and mass and one source galaxy, where we:

    # 1) Pass priors on the lens galaxy's light using the EllipticalSersic of the previous pipeline.
    # 2) Pass priors on the lens galaxy's  mass using the EllipticalIsothermal and ExternalShear fit of the previous
    #    pipeline.
    # 3) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.

    class LensSourcePhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            ### Lens Light, Sersic -> Sersic ###

            self.lens_galaxies.lens.light = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').\
                variable.lens_galaxies.lens.light

            ### Lens Mass, SIE -> PL ###

            self.lens_galaxies.lens.mass.centre = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').\
                variable_absolute(a=0.05).lens_galaxies.lens.mass.centre

            self.lens_galaxies.lens.mass.axis_ratio = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').\
                variable.lens_galaxies.lens.mass.axis_ratio

            self.lens_galaxies.lens.mass.phi = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').\
                variable.lens_galaxies.lens.mass.phi

            self.lens_galaxies.lens.mass.einstein_radius = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').\
                variable_absolute(a=0.3).lens_galaxies.lens.mass.einstein_radius

            ### Lens Shear, Shear -> Shear ###

            self.lens_galaxies.lens.shear = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').\
                variable.lens_galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.source_galaxies.source = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').\
                variable.source_galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            if pl_hyper_galaxies:
                self.lens_galaxies.lens.hyper_galaxy = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').hyper_galaxy. \
                    constant.lens_galaxies.lens.hyper_galaxy

                self.source_galaxies.source.hyper_galaxy = results.from_phase('phase_3_lens_sersic_sie_shear_source_sersic').hyper_galaxy. \
                    constant.source_galaxies.source.hyper_galaxy

    phase1 = LensSourcePhase(
        phase_name='phase_1_lens_sersic_pl_shear_source_sersic', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalPowerLaw,
                shear=mp.ExternalShear)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                light=lp.EllipticalSersic)),
        sub_grid_size=sub_grid_size, bin_up_factor=bin_up_factor, positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii, interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 75
    phase1.optimizer.sampling_efficiency = 0.2

    phase1 = phase1.extend_with_hyper_and_inversion_phases(
        hyper_galaxy=pl_hyper_galaxies)

    return pipeline.PipelineImaging(pipeline_name, phase1)