from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

# In this pipeline, we'll perform an analysis which initializes a lens model (the lens's light, mass and source's \
# light) and then fits the source galaxy using an inversion. This pipeline uses four phases:

# Phase 1:

# Description: Initializes the lens light model to subtract the foreground lens
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Description: Initializes the lens mass model and source light profile.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses the lens subtracted image from phase 1.

# Phase 3:

# Description: Refine the lens light and mass models and source light model.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens light (variable -> phase 1), lens mass and source light (variable -> phase 2).
# Notes: None

# Phase 4:

# Description: Initializes the source inversion parameters.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline).
# Notes: None

# Phase 5:

# Description: Refines the lens light and mass models using the source inversion.
# Lens Light: EllipticalSersic
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: AdaptiveMagnification + Constant
# Previous Pipelines: initializers/lens_sie_source_sersic_from_init.py
# Prior Passing: Lens Mass (variable -> previous pipeline), Source Inversion (constant -> phase 1)
# Notes: None

# ***NOTE*** Performing this analysis in a pipeline composed of 5 consectutive phases it not ideal, and it is better to
#            breaking the pipeline down into multiple pipelines. This is what is done in the 'pipelines/with_lens_light'
#            folder, using the pipelines:

#            1) initializers/lens_sersic_sie_source_sersic_from_init.py (phases 1->3)
#            2) initializers/lens_sersic_sie_source_inversion_from_pipeline.py (phases 4->5)

#            See runners/runner_adding_pipelines.py for more details on adding pipelines.

def make_pipeline(phase_folders=None):

    pipeline_name = 'pipeline_lens_sersic_sie_source_inversion'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    # We will switch between a circular mask which includes the lens light and an annular mask which removes it.

    def mask_function_circular(image):
        return msk.Mask.circular(shape=image.shape, pixel_scale=image.pixel_scale, radius_arcsec=3.0)

    def mask_function_annular(image):
        return msk.Mask.circular_annular(shape=image.shape, pixel_scale=image.pixel_scale,
                                         inner_radius_arcsec=0.3, outer_radius_arcsec=3.0)

    ### PHASE 1 ###

    # In phase 1, we will fit only the lens galaxy's light, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    # 2) Use a circular mask which includes the lens and source galaxy light.

    class LensPhase(ph.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light.centre_0 = prior.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.light.centre_1 = prior.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensPhase(phase_name='phase_1_lens_light_only', phase_folders=phase_folders,
                       lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                       optimizer_class=nl.MultiNest, mask_function=mask_function_circular)

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout the tutorial '' in howtolens/chapter_2).

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and source galaxy's light, where we:

    # 1) Use a lens-subtracted image generated by subtracting model lens galaxy image from phase 1.
    # 2) Use a circular annular mask which includes only the source-galaxy light.
    # 3) Initialize the priors on the centre of the lens galaxy's mass-profile by linking them to those inferred for \
    #    its light profile in phase 1.

    class LensSubtractedPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, results):
            return image - results.from_phase("phase_1_lens_light_only").unmasked_lens_plane_model_image

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass.centre_0 = \
                results.from_phase("phase_1_lens_light_only").variable.lens.light.centre_0
            self.lens_galaxies.lens.mass.centre_1 = \
                results.from_phase("phase_1_lens_light_only").variable.lens.light.centre_1

    phase2 = LensSubtractedPhase(phase_name='phase_2_lens_mass_and_source_light', phase_folders=phase_folders,
                                 lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                        shear=mp.ExternalShear)),
                                 source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                                 optimizer_class=nl.MultiNest, mask_function=mask_function_annular)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 60
    phase2.optimizer.sampling_efficiency = 0.2


    ### PHASE 3 ###

    # In phase 3, we will fit simultaneously the lens and source galaxies, where we:

    # 1) Initialize the lens's light, mass, shear and source's light using the results of phases 1 and 2.

    class LensSourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light = results.from_phase("phase_1_lens_light_only").variable.lens.light
            self.lens_galaxies.lens.mass = results.from_phase("phase_2_lens_mass_and_source_light").variable.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase("phase_2_lens_mass_and_source_light").variable.lens.shear
            self.source_galaxies.source = results.from_phase("phase_2_lens_mass_and_source_light").variable.source

    phase3 = LensSourcePhase(phase_name='phase_3_lens_light_mass_and_source_light', phase_folders=phase_folders,
                             lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                    mass=mp.EllipticalIsothermal,
                                                                    shear=mp.ExternalShear)),
                             source_galaxies=dict(source=gm.GalaxyModel(light=lp.EllipticalSersic)),
                             optimizer_class=nl.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    ### PHASE 4 ###

    # In phase 4, we initialize the inversion's resolution and regularization coefficient, where we:

    # 1) Use a lens-subtracted image generated by subtracting model lens galaxy image from phase 1.
    # 2) Fix our mass model to the lens galaxy mass-model from phase 2.
    # 3) Use a circular annular mask which includes only the source-galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def modify_image(self, image, results):
            return image - results.from_phase("phase_3_lens_light_mass_and_source_light").unmasked_lens_plane_model_image

        def pass_priors(self, results):

            self.lens_galaxies.lens.mass = results.from_phase("phase_3_lens_light_mass_and_source_light").constant.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase("phase_3_lens_light_mass_and_source_light").constant.lens.shear

    phase4 = InversionPhase(phase_name='phase_4_inversion_init', phase_folders=phase_folders,
                            lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            optimizer_class=nl.MultiNest, mask_function=mask_function_annular)

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    ### PHASE 5 ###

    # In phase 5, we fit the len galaxy light, mass and source galxy simultaneously, using an inversion. We will:

    # 1) Initialize the priors of the lens galaxy and source galaxy from phases 3+4.
    # 2) Use a circular mask which includes the lens and source galaxy light.

    class InversionPhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light = results.from_phase("phase_3_lens_light_mass_and_source_light").variable.lens.light
            self.lens_galaxies.lens.mass = results.from_phase("phase_3_lens_light_mass_and_source_light").variable.lens.mass
            self.lens_galaxies.lens.shear = results.from_phase("phase_3_lens_light_mass_and_source_light").variable.lens.shear
            self.source_galaxies.source = results.from_phase("phase_4_inversion_init").variable.source

    phase5 = InversionPhase(phase_name='phase_5_inversion', phase_folders=phase_folders,
                            lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic,
                                                                   mass=mp.EllipticalIsothermal,
                                                                   shear=mp.ExternalShear)),
                            source_galaxies=dict(source=gm.GalaxyModel(pixelization=pix.AdaptiveMagnification,
                                                                      regularization=reg.Constant)),
                            optimizer_class=nl.MultiNest, mask_function=mask_function_circular)

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 60
    phase5.optimizer.sampling_efficiency = 0.4

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4, phase5)