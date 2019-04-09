from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
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

# Whilst the 'nterp_pixel_scale can be manually specified in the pipeline, in this example we will make it
# an input parameter of the pipeline. This means we can run the pipeline with different pixel scales for different
# runners.

# We will also use phase tagging to ensure phases which use interpolation have a tag in their path, so it is clear
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

    pipeline_name = 'pipeline_interpolating_deflections'

    # This tag is 'added' to the phase path, to make it clear what binning up is used. The bin_up_tag and phase
    # name are shown for 3 example bin up factors:

    # interpolation_pixel_scale=None -> phase_path=phase_name
    # interpolation_pixel_scale=0.1 -> phase_path=phase_name_interp_0.1
    # interpolation_pixel_scale=0.05 -> phase_path=phase_name_interp_0.05

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return msk.Mask.circular_annular(shape=image.shape, pixel_scale=image.pixel_scale,
                                         inner_radius_arcsec=0.2, outer_radius_arcsec=3.3)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    class LensSourceX1Phase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass.centre_0 = prior.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.mass.centre_1 = prior.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(phase_name='phase_1_x1_source', phase_folders=phase_folders,
                               tag_phases=True,
                               lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                      shear=mp.ExternalShear)),
                               source_galaxies=dict(source_0=gm.GalaxyModel(light=lp.EllipticalSersic)),
                               mask_function=mask_function, optimizer_class=nl.MultiNest,
                               interp_pixel_scale=0.1)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Initialize the priors on the lens galaxy using the results of phase 1.
    # 2) Initialize the priors on the first source galaxy using the results of phase 1.

    class LensSourceX2Phase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_1_x1_source').variable.lens
            self.source_galaxies.source_0 = results.from_phase('phase_1_x1_source').variable.source_0

    phase2 = LensSourceX2Phase(phase_name='phase_2_x2_source', phase_folders=phase_folders,
                               tag_phases=True,
                               lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                      shear=mp.ExternalShear)),
                               source_galaxies=dict(source_0=gm.GalaxyModel(light=lp.EllipticalSersic),
                                                      source_1=gm.GalaxyModel(light=lp.EllipticalSersic)),
                               optimizer_class=nl.MultiNest, mask_function=mask_function,
                               interp_pixel_scale=interp_pixel_scale)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)