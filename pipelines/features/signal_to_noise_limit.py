import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp


# In this pipeline, we'll demonstrate signal-to-noise limiting - which allows us to fit ccd instrument where the noise-map
# has been increased to cap the maximum signal-to-noise value in the image in a phase of the pipeline. In this example,
# we will perform an initial analysis on an image with a signal-to-noise limit of 20.0, and then fit the image using
# the unscaled signal-to-noise map.

# Why would you want to limit the signal to noise? There are two reasons :
#
# 1) The model fitting, may be subject to  over-fitting the highest signal-to-noise regions of the image instead of
#    providing a global fit to the entire image. For example, if a lensed source has 4 really bright, compact, high
#    S/N images which are not fitted perfectly by the model, their high chi-squared contribution will drive the model
#    fit to place more light in those regions, ignoring the lensed source's lower S/N more extended arcs. Limiting the
#    S/N of these high S/N regions will reduce over-fitting. This is also important lens light subtractions which are
#    not perfect, which leave large chi-squared residuals due to the lens light's high S/N.

#    To learn more about this over-fitting problem, checkout chapter 5 of the 'HowToLens' lecture series.

# 2) If the model-fit has extremely large chi-squared values due to the high S/N of the instrument, this means the non-linear
#    search will take a long time mapping out this 'extreme' parameter space. In the early phases of a pipeline this
#    often isn't necessary, therefore a signal-to-noise limit can reduce the time an analysis takes to converge.

# Whilst signal to noise limits can be manually specified in the pipeline, in this example we will make the signal to
# noise limit an input parameter of the pipeline. This means we can run the pipeline with different signal to noise
# limits for different runners.

# We will also use phase tagging to ensure phases which use a signal to noise limit have a tag in their path, so it is
# clear what settings a phases has when it uses this feature.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Description: initialize the lens mass model and source light profile using x1 source with a signal to noise limit of 10.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses a signal to noise limit of 10

# Phase 2:

# Description: Fits the lens and source model using the true signal to noise map
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: No signal to noise limit.


def make_pipeline(phase_folders=None, signal_to_noise_limit=20.0):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_feature__signal_to_noise_limit"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings()

    # When a phase is passed a signal_to_noise_limit, a settings tag is automatically generated and added to the phase
    # path,to make it clear what signal-to-noise limit was used. The settings tag, phase name and phase paths are shown
    # for 3 example signal-to-noise limits:

    # signal_to_noise_limit=2 -> phase_path=phase_name/settings_snr_2
    # signal_to_noise_limit=3 -> phase_path=phase_name/settings_snr_3

    # If the signal_to_noise_limit is None, the tag is an empty string, thus not changing the settings tag:

    # signal_to_noise_limit=None -> phase_path=phase_name/settings

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag//'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return msk.Mask.circular_annular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.2,
            outer_radius_arcsec=3.3,
        )

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use a signal-to-noise limit of 20.0

    class LensSourceX1Phase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
            self.galaxies.lens.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(
        phase_name="phase_1__x1_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5, mass=mp.EllipticalIsothermal, shear=mp.ExternalShear
            ),
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
        ),
        mask_function=mask_function,
        signal_to_noise_limit=signal_to_noise_limit,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Omit the signal-to-noise limit, thus performing the modeling at the image's native signal-to-noise.

    class LensSourceX2Phase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens = results.from_phase(
                "phase_1__x1_source"
            ).variable.galaxies.lens

            self.galaxies.source = results.from_phase(
                "phase_1__x1_source"
            ).variable.galaxies.source

    phase2 = LensSourceX2Phase(
        phase_name="phase_2__x2_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5, mass=mp.EllipticalIsothermal, shear=mp.ExternalShear
            ),
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
        ),
        mask_function=mask_function,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)
