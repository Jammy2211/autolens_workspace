import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

# In this pipeline, we'll demonstrate deflection angle interpolation - which computes the deflection angles of a mass
# profile on a coarse 'interpolation grid' and then interpolates these values to the regular and sub-grids. The premise
# here is that for mass profiles that require computationally expensive numerical integration, this can reduce the
# number of integrations necessary from millions to thousands, giving speed-ups in the run times of order x100 or more!

# The interpolation grid is defined in terms of its pixel scale, and is automatically matched to the mask used in that
# phase. A higher resolution grid (i.e. lower pixel scale) will give more precise deflection angles, at the expense
# of longer calculation times. In this example we will use an interpolation pixel scale of 0.05", which in our 
# experiences balances run-time and precision. In the 'autolens_workspace/tools/precision' folder, you can find
# tools that allow you to experiment with the precision for different interpolation grids.

# Whilst the 'interp_pixel_scale can be manually specified in the pipeline, in this example we will make it
# an input parameter of the pipeline. This means we can run the pipeline with different pixel scales for different
# runners.

# We will also use settings tagging to ensure phases which use interpolation have a tag in their path, so it is clear
# that a phase uses this feature.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. We will use an elliptical power-law mass profile, instead of an isothermal ellipsoid, as 
# this profile requires numerical and integration and thus necessitates the use of interpolation to keep the run times
# manageable. This pipeline uses two phases:

# Phase 1:

# Description: Initializes the lens mass model and source light profile using x1 source with an interpolation pixel 
# scale of 0.1".
# Lens Mass: EllipitcalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses an interpolation pixel scale of 0.1"

# Phase 1:

# Description: Refine the lens model using a higher resolution interpolation grid.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: Uses an interpolation pixel scale of 0.05"

def make_pipeline(phase_folders=None, interp_pixel_scale=0.05):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pl__interpolating_deflections'

    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    # When a phase is passed a interp_pixel_scale, a settings tag is automatically generated and added to the phase path,
    # to make it clear what interpolation was used. The settings tag, phase name and phase paths are shown for 3
    # example bin up factors:

    # interpolation_pixel_scale=0.1 -> phase_path=phase_name/settings_interp_0.1
    # interpolation_pixel_scale=0.05 -> phase_path=phase_name/settings_interp_0.05

    # If the interp_pixel_scalel is None, the tag is an empty string, thus not changing the settings tag:

    # interpolation_pixel_scale=None -> phase_path=phase_name/settings

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    phase_folders.append(pipeline_name)

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return msk.Mask.circular_annular(
            shape=image.shape, pixel_scale=image.pixel_scale, inner_radius_arcsec=0.2, outer_radius_arcsec=3.3)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use an interpolation pixel scale of 0.1", to ensure fast deflection angle calculations.

    class LensSourceX1Phase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass.centre_0 = af.prior.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.mass.centre_1 = af.prior.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(
        phase_name='phase_1_x1_source', phase_folders=phase_folders, tag_phases=True,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear)),
        source_galaxies=dict(
            source_0=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic)),
        mask_function=mask_function, interp_pixel_scale=0.1,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Use the input interpolation pixel scale with (default) value 0.05", to ensure a more accurate modeling of the
    #    mass profile.

    class LensSourceX2Phase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_1_x1_source').\
                variable.lens_galaxies.lens

            self.source_galaxies.source_0 = results.from_phase('phase_1_x1_source').\
                variable.source_galaxies.source_0

    phase2 = LensSourceX2Phase(
        phase_name='phase_2_x2_source', phase_folders=phase_folders, tag_phases=True,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear)),
        source_galaxies=dict(
            source_0=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic),
            source_1=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic)),
        mask_function=mask_function,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)