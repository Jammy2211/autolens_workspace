import autofit as af
from autolens.data.array import mask as msk
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag

# This pipeline fits a strong lens which has two lens galaxies, and it is composed of the following 4 phases:

# Phase 1) Fit the light profile of the lens galaxy on the left of the image, at coordinates (0.0", -1.0").

# Phase 2) Fit the light profile of the lens galaxy on the right of the image, at coordinates (0.0", 1.0").

# Phase 3) Use this lens-subtracted image to fit the source galaxy's light. The mass-profiles of the two lens galaxies
#          can use the results of phases 1 and 2 to initialize their priors.

# Phase 4) Fit all relevant parameters simultaneously, using priors from phases 1, 2 and 3.

# Because the pipeline assumes the lens galaxies are at (0.0", -1.0") and (0.0", 1.0"), it is not a general pipeline
# and cannot be applied to any image of a strong lens.

def make_pipeline(phase_folders=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = 'pl__x2_lens_galaxies'

    pipeline_name = tag.pipeline_name_from_name_and_settings(pipeline_name=pipeline_name)

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'

    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # The left-hand galaxy is at (0.0", -1.0"), so we're going to use a small circular mask centred on its location to
    # fit its light profile. Its important that light from the other lens galaxy and source galaxy don't contaminate
    # our fit.

    def mask_function(image):
        return msk.Mask.circular(
            shape=image.shape, pixel_scale=image.pixel_scale,
            radius_arcsec=0.5, centre=(0.0, -1.0))

    class LeftLensPhase(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            # Lets restrict the prior's on the centres around the pixel we know the galaxy's light centre peaks.

            self.lens_galaxies.left_lens.light.centre_0 = \
                af.prior.GaussianPrior(mean=0.0, sigma=0.05)

            self.lens_galaxies.left_lens.light.centre_1 = \
                af.prior.GaussianPrior(mean=-1.0, sigma=0.05)

            # Given we are only fitting the very central region of the lens galaxy, we don't want to let a parameter 
            # like th Sersic index vary. Lets fix it to 4.0.

            self.lens_galaxies.left_lens.light.sersic_index = 4.0

    phase1 = LeftLensPhase(
        phase_name='phase_1_left_lens_light', phase_folders=phase_folders,
        lens_galaxies=dict(
            left_lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic)),
        mask_function=mask_function,
        optimizer_class=af.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5

    ### PHASE 2 ###

    # Now do the exact same with the lens galaxy on the right at (0.0", 1.0")

    def mask_function(image):
        return msk.Mask.circular(image.shape, pixel_scale=image.pixel_scale, radius_arcsec=0.5, centre=(0.0, 1.0))

    class RightLensPhase(phase_imaging.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.right_lens.light.centre_0 = \
                af.prior.GaussianPrior(mean=0.0, sigma=0.05)

            self.lens_galaxies.right_lens.light.centre_1 = \
                af.prior.GaussianPrior(mean=1.0, sigma=0.05)

            self.lens_galaxies.right_lens.light.sersic_index = 4.0

    phase2 = RightLensPhase(
        phase_name='phase_2_right_lens_light', phase_folders=phase_folders,
        lens_galaxies=dict(
            right_lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic)),
        mask_function=mask_function,
        optimizer_class=af.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 30
    phase2.optimizer.sampling_efficiency = 0.5

    ### PHASE 3 ###

    # In the next phase, we fit the source of the lens subtracted image.

    class LensSubtractedPhase(phase_imaging.LensSourcePlanePhase):

        # To modify the image, we want to subtract both the left-hand and right-hand lens galaxies. To do this, we need
        # to subtract the unmasked model image of both galaxies!

        def modify_image(self, image, results):

            phase_1_results = results.from_phase('phase_1_left_lens_light')
            phase_2_results = results.from_phase('phase_2_right_lens_light')

            return image - phase_1_results.unmasked_lens_plane_model_image - \
                   phase_2_results.unmasked_lens_plane_model_image

        def pass_priors(self, results):

            phase_1_results = results.from_phase('phase_1_left_lens_light')
            phase_2_results = results.from_phase('phase_2_right_lens_light')

            # We're going to link the centres of the light profiles computed above to the centre of the lens galaxy
            # mass-profiles in this phase. Because the centres of the mass profiles were fixed in phases 1 and 2,
            # linking them using the 'variable' attribute means that they stay constant (which for now, is what we want).

            self.lens_galaxies.left_lens.mass.centre_0 = \
                phase_1_results.variable.lens_galaxies.left_lens.light.centre_0

            self.lens_galaxies.left_lens.mass.centre_1 = \
                phase_1_results.variable.lens_galaxies.left_lens.light.centre_1

            self.lens_galaxies.right_lens.mass.centre_0 = \
                phase_2_results.variable.lens_galaxies.right_lens.light.centre_0

            self.lens_galaxies.right_lens.mass.centre_1 = \
                phase_2_results.variable.lens_galaxies.right_lens.light.centre_1

    phase3 = LensSubtractedPhase(
        phase_name='phase_3_fit_sources', phase_folders=phase_folders,
        lens_galaxies=dict(
            left_lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal),
            right_lens=gm.GalaxyModel(
                redshift=0.5,
                mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalExponential)),
        optimizer_class=af.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    ### PHASE 4 ###

    # In phase 4, we'll fit both lens galaxy's light and mass profiles, as well as the source-galaxy, simultaneously.

    class FitAllPhase(phase_imaging.LensSourcePlanePhase):

        def pass_priors(self, results):

            phase_1_results = results.from_phase('phase_1_left_lens_light')
            phase_2_results = results.from_phase('phase_2_right_lens_light')
            phase_3_results = results.from_phase('phase_3_fit_sources')

            # Results are split over multiple phases, so we setup the light and mass profiles of each lens separately.

            self.lens_galaxies.left_lens.light = phase_1_results.\
                variable.lens_galaxies.left_lens.light

            self.lens_galaxies.left_lens.mass = phase_3_results.\
                variable.lens_galaxies.left_lens.mass

            self.lens_galaxies.right_lens.light = phase_2_results.\
                variable.lens_galaxies.right_lens.light

            self.lens_galaxies.right_lens.mass = phase_3_results.\
                variable.lens_galaxies.right_lens.mass

            # When we pass a a 'variable' galaxy from a previous phase, parameters fixed to constants remain constant.
            # Because centre_0 and centre_1 of the mass profile were fixed to constants in phase 3, they're still
            # constants after the line after. We need to therefore manually over-ride their priors.

            self.lens_galaxies.left_lens.mass.centre_0 = phase_3_results.variable.\
                lens_galaxies.left_lens.mass.centre_0

            self.lens_galaxies.left_lens.mass.centre_1 = phase_3_results.variable.\
                lens_galaxies.left_lens.mass.centre_1

            self.lens_galaxies.right_lens.mass.centre_0 = phase_3_results.variable.\
                lens_galaxies.right_lens.mass.centre_0

            self.lens_galaxies.right_lens.mass.centre_1 = phase_3_results.variable.\
                lens_galaxies.right_lens.mass.centre_1

            # We also want the Sersic index's to be free parameters now, so lets change it from a constant to a
            # variable.

            self.lens_galaxies.left_lens.light.sersic_index = \
                af.prior.GaussianPrior(mean=4.0, sigma=2.0)

            self.lens_galaxies.right_lens.light.sersic_index = \
                af.prior.GaussianPrior(mean=4.0, sigma=2.0)

            # Things are much simpler for the source galaxies - just like them togerther!

            self.source_galaxies.source = phase_3_results.variable.source

    phase4 = FitAllPhase(
        phase_name='phase_4_fit_all', phase_folders=phase_folders,
        lens_galaxies=dict(
            left_lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal),
            right_lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                light=lp.EllipticalExponential)),
        optimizer_class=af.MultiNest)

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 60
    phase4.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4)
