import autofit as af
import autolens as al


# In this pipeline, we'll demonstrate sub-gridding, which defines the resolution of the sub-grid that is used to
# oversample the computation of image and deflection angles. In general, a higher level of sub-gridding provides
# numerically more precise results, at the expense of longer calculations and higher memory usage.

# Whilst sub grid sizes can be manually specified in the pipeline, in this example we will make the sub grid size
# an input parameter of the pipeline. This means we can run the pipeline with different sub grids for different
# runners.

# We will also use phase tagging to ensure phases which use different sub grid sizes have a different settings tag in
# their path, so it is clear what value a phase uses.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Description: initialize the lens mass model and source light profile using x1 source with a sub grid size of 2.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses a sub grid size of 2

# Phase 1:

# Description: Refine the lens and source model using a sub grid size of 4
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: Uses a sub grid size of 4.


def make_pipeline(phase_folders=None, sub_size=2):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline__feature"
    pipeline_tag = "sub_gridding"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    # When a phase is passed a sub_size, a settings tag is automatically generated and added to the phase path,
    # to make it clear what sub-grid was used. The settings tag, phase name and phase paths are shown for 3
    # example bin up factors:

    # sub_size=2 -> phase_path=phase_name/settings_sub_2
    # sub_size=3 -> phase_path=phase_name/settings_sub_3

    # If the sub-grid size is 1, the tag is an empty string, thus not changing the settings tag:

    # sub_size=1 -> phase_path=phase_name/settings

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(shape_2d, pixel_scales):
        return al.mask.circular_annular(
            shape_2d=shape_2d,
            pixel_scales=pixel_scales,
            inner_radius=0.2,
            outer_radius=3.3,
        )

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use a sub-grid size of 2x2 in every image pixel.

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__x1_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, mass=mass, shear=al.mp.ExternalShear),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        mask_function=mask_function,
        sub_size=2,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

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
                redshift=1.0, light=phase1.result.model.galaxies.source.light
            ),
        ),
        mask_function=mask_function,
        sub_size=sub_size,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2)
