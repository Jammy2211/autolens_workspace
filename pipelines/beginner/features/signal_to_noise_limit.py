import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we'll demonstrate signal-to-noise limiting - which fits data where the noise-map is increased to
# cap the highest signal-to-noise value. In this example, we will perform an initial analysis on an image with a
# signal-to-noise limit of 10.0, and then fit the image using the unscaled signal-to-noise map.

# Why would you want to limit the signal to noise?:

# 1) Model fitting may be subject to over-fitting the highest signal-to-noise regions of the image instead of
#    providing a global fit to the entire image. For example, if a lensed source has 4 really bright, compact, high
#    S/N images which are not fitted perfectly by the model, their high chi-squared contribution will drive the model
#    fit to place more light in those regions, ignoring the lensed source's lower S/N more extended arcs. Limiting the
#    S/N of these high S/N regions will reduce over-fitting. The same logic applies for foreground lens light
#    subtractions which are not perfect andn leave large chi-squared residuals.

#    To learn more about this over-fitting problem, checkout chapter 5 of the 'HowToLens' lecture series.

# 2) If the model-fit has extremely large chi-squared values due to the high S/N of the dataset. The non-linear
#    search will take a long time exploring this 'extreme' parameter space. In the early phases of a pipeline this
#    often isn't necessary, therefore a signal-to-noise limit can reduce the time an analysis takes to converge.

# The 'signal_to_noise_limit' is an input parameter of the pipeline, meaning we can run the pipeline with different
# limits using different runners.

# Phase names are tagged, ensuring phases using different signal to noise limits have a unique output path.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Fit the lens mass model and source light profile using x1 source with a signal to noise limit of 10.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: None
# Notes: Uses a signal to noise limit of 10

# Phase 2:

# Fit the lens and source model using the true signal to noise map

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: Lens mass (model -> phase 1), source light (model -> phase 1)
# Notes: No signal to noise limit.


def make_pipeline(phase_folders=None, signal_to_noise_limit=20.0):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__feature"
    pipeline_tag = "signal_to_noise_limit"

    # When a phase is passed a signal_to_noise_limit, a setup tag is automatically generated and added to the phase
    # path,to make it clear what signal-to-noise limit was used. The setup tag, phase name and phase paths are shown
    # for 3 example signal-to-noise limits:

    # signal_to_noise_limit=2 -> phase_path=phase_name/setup_snr_2
    # signal_to_noise_limit=3 -> phase_path=phase_name/setup_snr_3

    # If the signal_to_noise_limit is None, the tag is an empty string, thus not changing the setup tag:

    # signal_to_noise_limit=None -> phase_path=phase_name/setup

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag//'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use a signal-to-noise limit of 20.0

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__x1_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=mass, shear=al.mp.ExternalShear),
            source=al.GalaxyModel(redshift=1.0, sersic=al.lp.EllipticalSersic),
        ),
        signal_to_noise_limit=signal_to_noise_limit,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Omit the signal-to-noise limit, thus performing the modeling at the image's native signal-to-noise.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__x2_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=1.0, sersic=phase1.result.model.galaxies.source.sersic
            ),
        ),
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2)
