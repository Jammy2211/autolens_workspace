import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we'll demonstrate deflection angle interpolation - which computes the deflection angles of a mass
# profile on a coarse 'interpolation grid' and interpolates these values to the sub-grid. For mass profiles that require
# computationally expensive numerical integration, this reduces the number of integrations necessary from millions to
# thousands, giving speed-ups in the run times of over x100!

# The interpolation grid is defined in terms of a pixel scale and it is automatically matched to the mask used in that
# phase. A higher resolution grid (i.e. lower pixel scale) will give more precise deflection angles, at the expense
# of longer calculation times. In this example we will use an interpolation pixel scale of 0.05", which nicely
# balances run-time and precision. In the 'autolens_workspace/tools/precision' folder, you can find
# tools that allow you to experiment with the precision for different interpolation grids.

# The 'pixel_scale_interpolation_grid' is an input parameter of the pipeline, meaning we can run the pipeline
# with different interpolation grids using different runners.

# Phase names are tagged, ensuring phases using different interpolation grids have a unique output path.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. We will use a cored elliptical power-law mass profile, instead of an isothermal ellipsoid,
# as this profile requires expensive numerica integration.

# This pipeline uses two phases:

# Phase 1:

# Fit the lens mass model and source light profile using a source with an interpolation pixel scale of 0.1".

# Lens Mass: EllipticalPowerLaw + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: None
# Notes: Uses an interpolation pixel scale of 0.1"

# Phase 1:

# Refine the lens model using a higher resolution interpolation grid.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Prior Passing: Lens mass (model -> phase 1), source light (model -> phase 1)
# Notes: Uses an interpolation pixel scale of 0.05"


def make_pipeline(phase_folders=None, pixel_scale_interpolation_grid=0.05):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__feature"
    pipeline_tag = "interpolating_deflections"

    # When a phase is passed a 'pixel_scale_interpolation_grid', a setup tag is automatically generated and
    # added to the phase path to make it clear what interpolation was used. The setup tag, phase name and
    # phase paths are shown for 3 example 'pixel_scale_interpolation_grid' values:

    # interpolation_pixel_scale=0.1 -> phase_path=phase_name/setup_interp_0.1
    # interpolation_pixel_scale=0.05 -> phase_path=phase_name/setup_interp_0.05

    # If the pixel_scale_interpolation_grid is None, the tag is an empty string, thus not changing the setup tag:

    # interpolation_pixel_scale=None -> phase_path=phase_name/setup

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use an interpolation pixel scale of 0.1", to ensure fast deflection angle calculations.

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
        pixel_scale_interpolation_grid=0.1,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Use the input interpolation pixel scale with (default) value 0.05", to ensure a more accurate modeling of the
    #    mass profile.

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
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2)
