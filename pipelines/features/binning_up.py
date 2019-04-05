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

# In this pipeline, we'll demonstrate binning up - which allows us to fit a binned up version of an image in a phse of 
# the pipeline. In this example, we will perform an initial analysis on an image binned up from a pixel scale of 0.05" 
# to 0.01", which gives significant speed-up in run timme, and then refine the model in a second phase at the input
# resolution.

# Whilst bin up factors can be manually specified in the pipeline, in this example we will make the bin up factor
# an input parameter of the pipeline. This means we can run the pipeline with different binning up factors for different
# runners.

# We will also use phase tagging to ensure phases which use binned up data have a tag in their path, so it is clear
# that a phase uses this feature.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Description: Initializes the lens mass model and source light profile using x1 source with a bin up factor of x2.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses a bin up factor of x2

# Phase 1:

# Description: Fits the lens and source model using unbinned data.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: No binning up.

def make_pipeline(phase_folders=None, bin_up_factor=2):

    pipeline_name = 'pipeline_binning_up'

    # This tag is 'added' to the phase path, to make it clear what binning up is used. The bin_up_tag and phase
    # name are shown for 3 example bin up factors:

    # bin_up_factor=1 -> phase_path=phase_name
    # bin_up_factor=2 -> phase_path=phase_name_bin_up_factor_2
    # bin_up_factor=3 -> phase_path=phase_name_bin_up_factor_3

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
                               phase_tagging=True,
                               lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                      shear=mp.ExternalShear)),
                               source_galaxies=dict(source_0=gm.GalaxyModel(light=lp.EllipticalSersic)),
                               bin_up_factor=bin_up_factor,
                               mask_function=mask_function, optimizer_class=nl.MultiNest)

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
                               phase_tagging=True,
                               lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                      shear=mp.ExternalShear)),
                               source_galaxies=dict(source_0=gm.GalaxyModel(light=lp.EllipticalSersic),
                                                      source_1=gm.GalaxyModel(light=lp.EllipticalSersic)),
                               optimizer_class=nl.MultiNest, mask_function=mask_function)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)