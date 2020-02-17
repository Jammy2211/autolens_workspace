import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline we use sub-grids with different resoultions, that oversample the calculation of light profile
# intensities and mass profile deflection angles. In general, a higher level of sub-gridding provides numerically
# more precise results, at the expense of longer calculations and higher memory usage.

# The 'sub_size' is an input parameter of the pipeline, meaning we can run the pipeline with different binning up
# factors using different runners.

# Phase names are tagged, ensuring phases using different sub-sizes have a unique output path.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Fit the lens mass model and source light profile using x1 source with a sub grid size of 2.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: None
# Notes: Uses a sub grid size of 2

# Phase 1:

# Refine the lens and source model using a sub grid size of 4.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: Lens mass (model -> phase 1), source light (model -> phase 1)
# Notes: Uses a sub grid size of 4.


def make_pipeline(phase_folders=None, sub_size=2):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__feature"
    pipeline_tag = "sub_gridding"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    # When a phase is passed a 'sub_size,' a setup tag is automatically generated and added to the phase path,
    # to make it clear what sub-grid was used. The setup tag, phase name and phase paths are shown for 3
    # example sub_sizes:

    # sub_size=2 -> phase_path=phase_name/setup_sub_2
    # sub_size=3 -> phase_path=phase_name/setup_sub_3

    # If the sub-grid size is 1, the tag is an empty string, thus not changing the setup tag:

    # sub_size=1 -> phase_path=phase_name/setup

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use a sub-grid size of 2x2 in every image pixel.

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
        sub_size=2,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Use a sub-grid size of 4x4 in every image pixel.

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
        sub_size=sub_size,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2)
