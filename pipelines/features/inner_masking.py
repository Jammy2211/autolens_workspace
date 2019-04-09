from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
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

# We will use phase tagging to make it clear that the inner_mask_radii variable was used.

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

def make_pipeline(phase_folders=None, tag_phases=True, inner_mask_radii=None):

    pipeline_name = 'pipeline_inner_mask'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    # If phase tagging is on, a tag is 'added' to the phase name to make it clear what binning up is used. Phase names
    # with their tags are shown for 2 example inner mask radii values:

    # inner_mask_radii=0.2 -> phase_path='phase_name_inner_mask_0.20'
    # inner_mask_radii=0.25 -> phase_path='phase_name_inner_mask_0.25'

    # If the inner_mask_radii is None, the tag is an empty string, thus not changing the phase name:

    # inner_mask_radii=None -> phase_path='phase_name'

    ### PHASE 1 ###

    # In phase 1, we will:

    # 1) Specify a mask_function which uses a circular mask, but make the mask used in the analysis an annulus by
    #    specifying inner_mask_radii. This variable masks all pixels within its input radius. This is
    #    equivalent to using a circular_annular mask, however because it is a phase input variable we can turn this
    #    masking on and off for different phases.

    def mask_function(image):
        return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=2.5)

    phase1 = ph.LensSourcePlanePhase(phase_name='phase_1_use_inner_radii_input', tag_phases=tag_phases,
                                     phase_folders=phase_folders,
                                     lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest, mask_function=mask_function,
                                     inner_mask_radii=inner_mask_radii)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will:

    # 1) Initialize the priors on the lens galaxy's mass and source galaxy's light by linking them to those inferred
    #    in phase 1.

    # 2) Not specify a mask function to the phase, meaning that by default the custom mask passed to the pipeline when
    #    we run it will be used instead.

    class LensSubtractedPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_1_use_inner_radii_input').variable.lens
            self.source_galaxies.source = results.from_phase('phase_1_use_inner_radii_input').variable.source

    phase2 = LensSubtractedPhase(phase_name='phase_2_circular_mask', phase_folders=phase_folders,
                                 tag_phases=True,
                                 lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.2

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)