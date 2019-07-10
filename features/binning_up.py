import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

# In this pipeline, we'll demonstrate binning up - which allows us to fit a binned up version of an image in a phse of 
# the pipeline. In this example, we will perform an initial analysis on an image binned up from a pixel scale of 0.05" 
# to 0.1", which gives significant speed-up in run timme, and then refine the model in a second phase at the input
# resolution.

# Whilst bin up factors can be manually specified in the pipeline, in this example we will make the bin up factor
# an input parameter of the pipeline. This means we can run the pipeline with different binning up factors for different
# runners.

# We will also use phase tagging to ensure phases which use binned up data have a tag in their path, so it is clear
# what settings a phases has when it uses this feature.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Description: Initializes the lens mass model and source light profile using x1 source with a bin up factor of x2.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses a bin up factor of x2

# Phase 2:

# Description: Fits the lens and source model using unbinned data.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: No binning up.

def make_pipeline(phase_folders=None, bin_up_factor=2):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pl__binning_up'

    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    # When a phase is passed a bin_up_factor, a settings tag is automatically generated and added to the phase path,
    # to make it clear what binning up was used. The settings tag, phase name and phase paths are shown for 3
    # example bin up factors:

    # bin_up_factor=2 -> phase_path=phase_name/settings_bin_up_2
    # bin_up_factor=3 -> phase_path=phase_name/settings_bin_up_3

    # If the bin_up_facor is None or 1, the tag is an empty string, thus not changing the settings tag:

    # bin_up_factor=None -> phase_path=phase_name/settings
    # bin_up_factor=1 -> phase_path=phase_name/settings

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/settings_tag/'

    phase_folders.append(pipeline_name)

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return msk.Mask.circular_annular(shape=image.shape, pixel_scale=image.pixel_scale,
                                         inner_radius_arcsec=0.2, outer_radius_arcsec=3.3)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Bin up the image by the input factor specified, which is default 2.

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
        mask_function=mask_function, bin_up_factor=bin_up_factor,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Omit the bin up factor, thus performing the modeling at the image's native resolution.

    class LensSourceX2Phase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_1_x1_source').\
                variable.lens_galaxies.lens

            self.source_galaxies.source = results.from_phase('phase_1_x1_source').\
                variable.source_galaxies.source

    phase2 = LensSourceX2Phase(
        phase_name='phase_2_x2_source', phase_folders=phase_folders, tag_phases=True,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal,
                shear=mp.ExternalShear)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic)),
        mask_function=mask_function,
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)