import autofit as af
import autolens as al

### PIPELINE DESCRIPTION ###

# In this pipeline, we'll demonstrate the use of an inversion pixel limit - which places an upper limit on the number of
# pixels an inversion uses in a pipeline and thus speeding up run time. In this example, we will perform an initial
# analysis which is restricted to just 100 pixels and then refine the model in a second phase using a higher resolution.

# The 'inversion_pixel_limit' is an input parameter of the pipeline, meaning we can run the pipeline with different
# pixel limits using different runners.

# Phase names are tagged, ensuring phases using different bin factors have a unique output path.

# We'll perform a basic analysis which fits a lensed source galaxy using an inversion where
# the lens's light is omitted.

# This pipeline uses two phases:

# Phase 1:

# Fit the lens mass model and source light profile using an inversion with a pixel limit.

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification
# Prior Passing: None
# Notes: Uses an inversion pixel limit of 100

# Phase 2:

# Fit the lens and source model without an inversion pixel limit

# Lens Mass: EllipticalIsothermal + ExternalShear
# Source Light: VoronoiMagnification
# Prior Passing: Lens mass (model -> phase 1), source light (model -> phase 1)
# Notes: No pixel limit


def make_pipeline(phase_folders=None, inversion_pixel_limit=100):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline__feature"
    pipeline_tag = "inversion_pixel_limit"

    # When a phase is passed an 'inversion_pixel_limit', a setup tag is automatically generated and added to the
    # phase path to make it clear what limit was used. The setup tag, phase name and phase paths are shown for 3
    # example inversion pixel limits:

    # inversion_pixel_limit=50 -> phase_path=phase_name/setup_pix_lim_50
    # inversion_pixel_limit=80 -> phase_path=phase_name/setup_pix_lim_80

    # If the inversion pixel limit is None, the tag is an empty string, thus not changing the setup tag:

    # inversion_pixel_limit=None -> phase_path=phase_name/setup
    # inversion_pixel_limit=1 -> phase_path=phase_name/setup

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag//'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Limit the number of source pixels used by the inversion.

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__x1_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=mass, shear=al.mp.ExternalShear),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        inversion_pixel_limit=inversion_pixel_limit,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Omit the inversion pixel limit, thus performing the modeling at a high source plane resolution if necessary.

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
                redshift=1.0,
                pixelization=phase1.result.model.galaxies.source.pixelization,
                regularization=phase1.result.model.galaxies.source.regularization,
            ),
        ),
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2)
