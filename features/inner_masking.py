import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

# In this pipeline, we show how custom masks (generated using the tools/mask_maker.py file) can be input and used by a
# pipeline. We will also use the inner_mask_radii variable of a phase to mask the central regions of an image.

# The inner circular mask radii will be passed to the pipeline as an input parameter, such that it can be customizmed
# in the runner. The first phase will use the a circular mask function with a circle of radius 3.0", but mask the
# central regions of the image via the inner_mask_radii input variable.

# We will use settings tagging to make it clear that the inner_mask_radii variable was used.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Description: Fits the lens and source model using an annular mask.
# Lens Mass: EllipitcalIsothermal
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses a mask where the inner regions are masked via the inner_mask_radii parameter.

# Phase 1:

# Description: Fits the lens and source model using a circular mask.
# Lens Mass: EllipitcalIsothermal
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: Uses a circular mask.

# Checkout the 'workspace/runners/pipeline_runner.py' script for how the custom mask and positions are loaded and used
# in the pipeline.

def make_pipeline(phase_folders=None, inner_mask_radii=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pl__inner_mask'

    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    # When a phase is passed an inner_mask_radii, a settings tag is automatically generated and added to the phase
    # path to make it clear what mask_radii was used. The settings tag, phase name and phase paths are shown for 2
    # example inner mask radii:

    # inner_mask_radii=0.2 -> phase_path='phase_name/settings_inner_mask_0.20'
    # inner_mask_radii=0.25 -> phase_path='phase_name/settings_inner_mask_0.25'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    # If the inner_mask_radii is None, the tag is an empty string, thus not changing the settings tag:

    # inner_mask_radii=None -> phase_path='phase_name/settings'

    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we will:

    # 1) Specify a mask_function which uses a circular mask, but makes the mask used in the analysis an annulus by
    #    specifying an inner_mask_radii. This variable masks all pixels within this input radius. This is
    #    equivalent to using a circular_annular mask, however because it is a phase input variable we can turn this
    #    masking on and off for different phases.

    def mask_function(image):
        return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=2.5)

    phase1 = phase_imaging.LensSourcePlanePhase(
        phase_name='phase_1_use_inner_radii_input', phase_folders=phase_folders, tag_phases=True,
        lens_galaxies=dict(lens=gm.GalaxyModel(
            redshift=0.5,
            mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(source=gm.GalaxyModel(
            redshift=1.0,
            light=lp.EllipticalSersic)),
        mask_function=mask_function, inner_mask_radii=inner_mask_radii,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will:

    # 1) Not specify a mask function to the phase, meaning that by default the custom mask passed to the pipeline when
    #    we run it will be used instead.

    class LensSubtractedPhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_1_use_inner_radii_input').\
                variable.lens_galaxies.lens

            self.source_galaxies.source = results.from_phase('phase_1_use_inner_radii_input').\
                variable.source_galaxies.source

    phase2 = LensSubtractedPhase(
        phase_name='phase_2_circular_mask', phase_folders=phase_folders, tag_phases=True,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.2

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)