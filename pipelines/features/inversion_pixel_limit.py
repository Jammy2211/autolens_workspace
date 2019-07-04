import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

# In this pipeline, we'll demonstrate the use of an inversion pixel limit - which allows us to put an upper limit on 
# the number of pixels an inversion uses in a pipeline. In this example, we will perform an initial analysis which is 
# restricted to just 100 pixels, for super fast run-times, and then refine the model in a second phase at the input
# resolution.

# Whilst the inversion pixel limit can be manually specified in the pipeline, in this example we will make the limit
# an input parameter of the pipeline. This means we can run the pipeline with different pixel limits for different
# runners.

# We will also use phase tagging to ensure phases which use a pixel limit have a tag in their path, so it is clear
# what settings a phases has when it uses this feature.

# We'll perform a basic analysis which fits a lensed source galaxy using an inversion where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Description: Initializes the lens mass model and source light profile using an inversion with a pixel limit.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses an inversion pixel limit of 100

# Phase 2:

# Description: Fits the lens and source model without an inversion pixel limit
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: VoronoiMagnification
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: No pixel limit

def make_pipeline(phase_folders=None, inversion_pixel_limit=100):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pl__inversion_pixel_limit'

    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    # When a phase is passed a inversion_pixel_limit, a settings tag is automatically generated and added to the 
    # phase path to make it clear what binning up was used. The settings tag, phase name and phase paths are shown for 3
    # example inversion pixel limits:

    # inversion_pixel_limit=50 -> phase_path=phase_name/settings_pix_lim_50
    # inversion_pixel_limit=80 -> phase_path=phase_name/settings_pix_lim_80

    # If the inversion pixel limit is None, the tag is an empty string, thus not changing the settings tag:

    # inversion_pixel_limit=None -> phase_path=phase_name/settings
    # inversion_pixel_limit=1 -> phase_path=phase_name/settings

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/settings_tag/'

    phase_folders.append(pipeline_name)

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return msk.Mask.circular_annular(
            shape=image.shape, pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.2, outer_radius_arcsec=3.3)

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Limit the number of source pixels used by the inversion.

    class LensSourceX1Phase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass.centre_0 = \
                af.prior.GaussianPrior(mean=0.0, sigma=0.1)

            self.lens_galaxies.lens.mass.centre_1 = \
                af.prior.GaussianPrior(mean=0.0, sigma=0.1)

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
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant)),
        mask_function=mask_function, inversion_pixel_limit=inversion_pixel_limit,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Omit the inversion pixel limit, thus performing the modeling at a high source plane resolution if necessary.

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
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant)),
        mask_function=mask_function,
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)