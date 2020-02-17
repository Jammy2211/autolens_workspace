import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we'll demonstrate binning up - which allows us to fit a binned up version of an image in a phse of
# the pipeline. In this example, we will perform an initial analysis on an image binned up from a pixel scale of 0.05"
# to 0.1", which gives significant speed-up in run time. We'll then refine the model in a second phase at the native
# resolution.

# The 'bin up factor' is an input parameter of the pipeline, meaning we can run the pipeline with different binning up
# factors using different runners.

# Phase names are tagged, ensuring phases using different bin factors have a unique output path.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Fit the lens mass model and source light profile using x1 source with a bin up factor of x2.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: None
# Notes: Uses a bin up factor of x2

# Phase 2:

# Fit the lens and source model using unbinned dataset.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: Lens mass (model -> phase 1), source light (model -> phase 1)
# Notes: No binning up.


def make_pipeline(phase_folders=None, bin_up_factor=2):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__feature"
    pipeline_tag = "binning_up"

    # When a phase is passed a 'bin_up_factor', a setup tag is automatically generated and added to the phase path,
    # to make it clear what binning up was used. The setup tag, phase name and phase paths are shown for 3
    # example bin up factors:

    # bin_up_factor=2 -> phase_path=phase_name/setup_bin_2
    # bin_up_factor=3 -> phase_path=phase_name/setup_bin_3

    # If the bin_up_factor is None or 1, the tag is an empty string, thus not changing the setup tag:

    # bin_up_factor=None -> phase_path=phase_name/setup
    # bin_up_factor=1 -> phase_path=phase_name/setup

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag//'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Bin up the image by the input factor specified, which is default 2.

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
        bin_up_factor=bin_up_factor,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Omit the bin up factor, thus performing the modeling at the image's native resolution.

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
