import autofit as af
import autolens as al

# This pipeline fits a strong lens which has two lens galaxies. It is composed of the following 4 phases:

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
    # will be the string specified below. However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline__x2_galaxies"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # Let's restrict the priors on the centres around the pixel we know the galaxy's light centre peaks.

    left_lens = al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)

    left_lens.light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)

    left_lens.light.centre_1 = af.GaussianPrior(mean=-1.0, sigma=0.05)

    # Given we are only fitting the very central region of the lens galaxy, we don't want to let a parameter
    # like the Sersic index vary. Lets fix it to 4.0.

    left_lens.light.sersic_index = 4.0

    phase1 = al.PhaseImaging(
        phase_name="phase_1__left_lens_light",
        phase_folders=phase_folders,
        galaxies=dict(
            left_lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)
        ),
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5
    phase1.optimizer.evidence_tolerance = 100.0

    ### PHASE 2 ###

    # Now do the exact same with the lens galaxy on the right at (0.0", 1.0").

    # We will additionally pass the left lens's light model as an instance, which as we learnted in the tutorial means
    # its included in the model with fixed parameters.

    right_lens = al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)

    right_lens.light.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.05)

    right_lens.light.centre_1 = af.GaussianPrior(mean=1.0, sigma=0.05)

    right_lens.light.sersic_index = 4.0

    phase2 = al.PhaseImaging(
        phase_name="phase_2__right_lens_light",
        phase_folders=phase_folders,
        galaxies=dict(
            left_lens=phase1.result.instance.galaxies.left_lens, right_lens=right_lens
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 30
    phase2.optimizer.sampling_efficiency = 0.5
    phase2.optimizer.evidence_tolerance = 100.0

    ### PHASE 3 ###

    # In the next phase, we fit the source of the lens subtracted image. We will of course use fixed lens light
    # models for both the left and right lens galaxies.

    # We're going to link the centres of the light profiles computed above to the centre of the lens galaxy
    # mass-profiles in this phase. Because the centres of the mass profiles were fixed in phases 1 and 2,
    # linking them using the 'variable' attribute means that they stay constant (which for now, is what we want).

    left_lens = al.GalaxyModel(
        redshift=0.5,
        light=phase1.result.instance.galaxies.left_lens.light,
        mass=al.mp.EllipticalIsothermal,
    )
    right_lens = al.GalaxyModel(
        redshift=0.5,
        light=phase2.result.instance.galaxies.right_lens.light,
        mass=al.mp.EllipticalIsothermal,
    )

    left_lens.mass.centre_0 = phase1.result.model.galaxies.left_lens.light.centre_0

    left_lens.mass.centre_1 = phase1.result.model.galaxies.left_lens.light.centre_1

    right_lens.mass.centre_0 = phase2.result.model.galaxies.right_lens.light.centre_0

    right_lens.mass.centre_1 = phase2.result.model.galaxies.right_lens.light.centre_1

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_x2_sie__source_exp",
        phase_folders=phase_folders,
        galaxies=dict(
            left_lens=left_lens,
            right_lens=right_lens,
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalExponential),
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5
    phase3.optimizer.evidence_tolerance = 100.0

    ### PHASE 4 ###

    # In phase 4, we'll fit both lens galaxy's light and mass profiles, as well as the source-galaxy, simultaneously.

    # Results are split over multiple phases, so we setup the light and mass profiles of each lens separately.

    left_lens = al.GalaxyModel(
        redshift=0.5,
        light=phase1.result.model.galaxies.left_lens.light,
        mass=phase3.result.model.galaxies.left_lens.mass,
    )

    right_lens = al.GalaxyModel(
        redshift=0.5,
        light=phase2.result.model.galaxies.right_lens.light,
        mass=phase3.result.model.galaxies.right_lens.mass,
    )

    # When we pass a a 'model' galaxy from a previous phase, parameters fixed to constants remain constant.
    # Because centre_0 and centre_1 of the mass profile were fixed to constants in phase 3, they're still
    # constants after the line after. We need to therefore manually over-ride their priors.

    left_lens.mass.centre_0 = phase3.result.model.galaxies.left_lens.mass.centre_0

    left_lens.mass.centre_1 = phase3.result.model.galaxies.left_lens.mass.centre_1

    right_lens.mass.centre_0 = phase3.result.model.galaxies.right_lens.mass.centre_0

    right_lens.mass.centre_1 = phase3.result.model.galaxies.right_lens.mass.centre_1

    # We also want the Sersic index's to be free parameters now, so lets change it from a constant to a
    # variable.

    left_lens.light.sersic_index = af.GaussianPrior(mean=4.0, sigma=2.0)

    right_lens.light.sersic_index = af.GaussianPrior(mean=4.0, sigma=2.0)

    # Things are much simpler for the source galaxies - just link them together!

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_x2_sersic_sie__source_exp",
        phase_folders=phase_folders,
        galaxies=dict(
            left_lens=left_lens,
            right_lens=right_lens,
            source=phase3.result.model.galaxies.source,
        ),
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 60
    phase4.optimizer.sampling_efficiency = 0.5

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
