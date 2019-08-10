# so, you're probably wondering, how does hyper-mode work in the context of pipelines? Afterall, there are a lot more
# things we have to do now; pass hyper-images between phases, use different pixelizations and regularization schemes,
# and decide what parts of the noise-map we do and don't want to scale.

### HYPER IMAGE PASSING ###

# So, lets start with hyper-image passing. That is, how do we pass the model images of the lens and source galaxies
# to later phases, so they can be used as hyper-images? Well, I've got some good news, *we no nothing*. We've designed
# PyAutoLens to automatically pass hyper-images between phases, so you don't have to! Want use these features? Just
# attach the classes to your GalaxyModel's like you would the light and mass models, nice!

# However, PyAutoLens does need to know which model images we pass to which galaxies between phases. To do this,
# PyAutoLens uses galaxy-names. When you create a GalaxyModel, you name the galaxies, for example below we've called
# the lens galaxy 'lens' and the source galaxy 'source':

# phase2 = phase_imaging.LensSourcePlanePhase(
#     phase_name="phase_1",
#     phase_folders=phase_folders,
#     galaxies=dict(
# NAME --> lens=gm.GalaxyModel(
#             redshift=0.5,
#             light=lp.EllipticalSersic,
#             mass=mp.EllipticalIsothermal,
#         )
#     ),
#       galaxies=dict(
# NAME --> source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic)
#     ),
# )

# To pass the resulting model images of these galaxies to galaxies in the next phase galaxies are paired by their names.
# So, the lens galaxy would need to be called 'lens' again, and the source galaxy 'source' again. This is probably the
# naming convention you've followed using PyAutoLens up to now anyway, so this should come naturually.
#
# The reason we use this galaxy naming scheme is for more complex lens systems, which may have multiple lens and source
# galaixes. In these cases, it is important that we can link galaxies together by explicitly, using their names.

### HYPER PHASES ###

# There is only one more big change to hyper-pipelines from regular pipelines. That is, how do we fit for the
# hyper-parameters using our non-linear search (e.g. MultiNest)? What we don't want to do is fit for them all whilst
# also fitting the lens and source models. That would create an unnecessarily large paameter space, which we'd fail to
# fit accurately and efficiently.

# So, we instead 'extend' phases with extra phases that specifically fit certain components of hyper-mode. You've
# hopefully already seen this with the following line of code, which optimizes the hyper-parameters of just the
# inverison (e.g. the pixelization and regularization):

# phase1 = phase1.extend_with_inversion_phase()

# Extending a phase with hyper phases is just as easy:

# phase1 = phase1.extend_with_multiple_hyper_phases(
#     hyper_galaxy=True,
#     include_background_sky=True,
#     include_background_noise=True,
#     inversion=True
# )

# This extends te phase with 3 addition phases, which do the following:

# 1) Simultaneously fit all hyper-galaxies, the background sky and background noise hyper parameters, using the best-fit
#    lens and source models from the normal phase. Thus, this phase only scales the noise and the image. This is called
#    the 'hyper_galaxy' phase.

# 2) Fit all of the inversion parameters, using the pixelization and regularization scheme that were used in the normal
#    phase. Typically, this would be a brightness-based pixelization and adaptive regularization scheme, but if it were
#    magnification based and constant it would work just fine. Again, the previous best-fit lens and source models are
#    used. This is called the 'inversion' phase.

# 3) Fit all of the components above, using Gaussian priors centred on the resulting values in steps 1) and 2). This is
#    important as there is trade-off between increasing the noise in the lens / source and changing the pixelization /
#    regularization hyper-parameters. This is called the 'hyper_combined' phase.

# In the pipeline below, you'll see we use the results of these phases (typically just the 'hyper_combined' phase) to
# setup the hyper-galaxies, hyper-instrument, pixelization and regularization in the next phase. Infact, in this pipeline
# all these components will then be setup as 'constants' and therefore fixed during each phases 'normal' optimization
# that changes lens and source models.

### HYPER MASKING ###

# There is one more rule you should know for hyper-mode. That is, the mask *cannot* change between phases. This is
# because of hyper-image passing. Basically, if you were to change a mask such that it adds new, extra pixels that
# were not modeled in the previous phase, these pixels wouldn't be contained in the hyper-image. This causes some
# very nasty problems. So, instead, we simply ask that you use the same mask throughout the analysis. To ensure
# this happens, for hyper-pipelines we require that the mask is passed to the 'pipeline.run' function.

import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

# Okay, so for our example hyper-pipeline, we're going to run 7 (!) phases. Conceptually, there isn't too much new
# here compared to regular pipelines. But, we need so many phases to explain a few design choices that arise for
# hyper pipelines. An overview of the pipeline is as follows:

# Phase 1) Fit and subtract the lens galaxies light profile.
# Phase 1 Extension) Fit the lens galaxy hyper galaxy and background noise.

# Phase 2) Fit the lens galaxy mass model and source light profile, using the lens subtracted image, the lens
#          hyper-galaxy noise map and background noise from phase 1.
# Phase 2 Extension) Fit the lens / source hyper galaxies and background noise.

# Phase 3) Fit the lens light, mass and source using priors from phases 1 & 2, and using the hyper lens / source
#          galaxies from phase 2.
# Phase 3 Extension) Fit the lens / source hyper galaxies and background noise.

# Phase 4) Initialize an inversion of the source-galaxy, using a magnification based pixelization and constant
#          regularization scheme.
# Phase 4 Extension) Fit the lens / source hyper galaxies and background noise)

# Phase 5) Refine the lens light and mass models using the inversion from phase 4.
# Phase 5 Extension) Fit the lens / source hyper galaxies and background noise.

# Phase 6) Initialize an inversion of the source-galaxy, using a brightness based pixelization and adaptive
#          regularization scheme.
# Phase 6 Extension) Fit the lens / source hyper galaxies, background noise and reoptimize the inversion.

# Phase 7) Refine the lens light and mass models using the inversion from phase 6.
# Phase 7 Extension) Fit the lens / source hyper galaxies, background noise and reoptimize the inversion.

# Phew! Thats a lot of phases, so lets take a look.

# For hyper-pipelines, we pass bools into the function to determine whether certain hyper features are on or off. For
# example, if pipeline_settings.hyper_galaxies is False, then noise-scaling via hyper-galaxies will be omitted from this run of the
# pipeline. We also pass in the pixelization / regularization schemes that will be used in phase 6 and 7 of the
# pipeline.


def make_pipeline(
    pipeline_settings,
    phase_folders=None,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # The pipeline name, tagging and phase folders work exactly like they did in previous phases. However, tagging now
    # also includes the pixelization and regularization schemes, as these cannot be changed foro a hyper-pipeline.

    pipeline_name = "pipeline__hyper_example"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings(
        pixelization=pipeline_settings.pixelization, regularization=pipeline_settings.regularization
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)
    
    ### SETUP SHEAR ###

    # If the pipeline should include shear, add this class below so that it enters the phase.

    # After this pipeline this shear class is passed to all subsequent pipelines, such that the shear is either
    # included or omitted throughout the entire pipeline.

    if pipeline_settings.include_shear:
        shear = mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # We set up and run phase 1 as per usual for regular pipelines, so nothing new here.

    phase1 = phase_imaging.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=gm.GalaxyModel(redshift=0.5, light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    # This extends phase 1 with hyper-phases that fit for the hyper-galaxies, as described above. The extension below
    # adds two phases, a 'hyper-galaxy' phase which fits for the lens hyper galaxy + the background noise, and a
    # 'hyper_combined' phase which fits them again.
    #
    # Although this might sound like unnecessary repetition, the second phase uses Gaussian priors inferred from the
    # first phase, meaning that it can search regions of parameter space that may of been unaccessible due to the
    # first phase's uniform piors.

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 2 ###

    # This phase fits for the lens's mass model and source's light profile, using the lens subtracted image from phase
    # 1. The lens galaxy hyper-galaxy will be included, such that high chi-squared values in the central regions of the
    # image due to a poor lens light subtraction are reduced by noise scaling and do not significantly impact the fit.

    class LensSubtractedPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            # Most of this pass priors function is the usual, e.g. passing the lens galaxy's light profile from phase
            # 1 to phase 2.

            ## Lens Light Sersic -> Sersic ##

            self.galaxies.lens.light = results.from_phase(
                "phase_1__lens_sersic"
            ).constant.galaxies.lens.light

            ## Lens Mass, Move centre priors to centre of lens light ###

            self.galaxies.lens.mass.centre = (
                results.from_phase("phase_1__lens_sersic")
                .variable_absolute(a=0.1)
                .galaxies.lens.light.centre
            )

            # These are new to a hyper-pipeline, and will appear in every pass priors function. Basically, these
            # check whether a hyper-feature is turned on. If it is, then it will have been fitted for in the previous
            # phase's 'hyper_combined' phase, so its parameters are passed to this phase as constants.

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    # This extends phase 2 with hyper-phases that fit for the hyper-galaxies, as described above. This extension again
    # adds two phases, a 'hyper_galaxy' phase and 'hyper_combined' phase. Unlike the extension to phase 1 which only
    # include a lens hyper-galaxy, this extension includes both the lens and source hyper galaxies, as well as the
    # background noise.

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 3 ###

    # Usual stuff in this phase - we fit the lens and source using priors from the above 2 phases.

    class LensSourcePhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):
            ## Lens Light, Sersic -> Sersic ###

            self.galaxies.lens.light = results.from_phase(
                "phase_1__lens_sersic"
            ).variable.galaxies.lens.light

            ## Lens Mass, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens.mass = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.lens.mass

            self.galaxies.lens.shear = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.lens.shear

            ### Source Light, Sersic -> Sersic ###

            self.galaxies.source = results.from_phase(
                "phase_2__lens_sie__source_sersic"
            ).variable.galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            # Although the above hyper phase includes fitting for the source galaxy, at this early stage in the pipeline
            # we make a choice not to pass the hyper-galaxy of the source. Why? Because there is a good chance
            # our simplistic single Sersic profile won't yet provide a good fit to the source.
            #
            # If this is the case, it scaled noise map won't be very good. It isn't until we are fitting the
            # source using an inversion that we begin to pass its hyper galaxy, e.g. when we can be confident our fit
            # to the instrument is reliable!

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase3 = LensSourcePhase(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic),
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    # The usual phase extension, which operates the same as the extension for phase 2.

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 4 ###

    # Next, we initialize a magnification based pixelization and constant regularization scheme. But, you're probably
    # wondering, why do we bother with these at all? Why not jump straight to the brightness based pixelization and
    # adaptive regularization?

    # Well, its to do with the hyper-images of our source. At the end of phase 3, we've only fitted the source galaxy
    # using a single EllipticalSersic profile. What if the source galaxy is more complex than a Sersic? Or has
    # multiple components? Our fit, put simply, won't be very good! This makes for a bad hyper-image.

    # So, its beneficial for us to introduce an intermediate inversion using a magnification based grid, that will fit
    # all components of the source accurately, so that we have a good quality hyper image for the brightness based
    # pixelization and adaptive regularization. Its for this reason we've also omitted the hyper source galaxy from
    # the phases above; if the hyper-image were poor, so is the scaled noise-map!

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).constant.galaxies.lens

            ## Set all hyper-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase4 = InversionPhase(
        phase_name="phase_4__source_inversion_initialize_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    # This is the usual phase extensions. Given we're only using this inversion to refine our hyper-images, we won't
    # bother reoptimizing its hyper-parameters

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 5 ###

    # Here, we refine the lens light and mass models using this magnification based pixelization and constant
    # regularization scheme. If our source model was a bit dodgy because it was more complex than a single Sersic, it
    # probably means our lens model was too, so a quick refinement in phase 5 is worth the effort!

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_4__source_inversion_initialize_magnification"
            ).constant.galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase5 = InversionPhase(
        phase_name="phase_5__lens_sersic_sie__source_inversion_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_settings.pixelization,
                regularization=pipeline_settings.regularization,
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 75
    phase5.optimizer.sampling_efficiency = 0.2

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 6 ###

    # In phase 6, we finally use our hyper-mode features to adapt the pixelization to the source's morphology
    # and the regularization to its brightness! This phase works like a normal initialization phase, whereby we fix
    # the lens and source models to the best-fit of the previous phase and just optimize the hyper-parameters.

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_5__lens_sersic_sie__source_inversion_magnification"
            ).constant.galaxies.lens

            ## Set all hyper-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase6 = InversionPhase(
        phase_name="phase_6__source_inversion_initialize",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_settings.pixelization,  # <- This is our brightness based pixelization provided it was input into the pipeline.
                regularization=pipeline_settings.regularization,  # <- And this our adaptive regularization.
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase6.optimizer.const_efficiency_mode = True
    phase6.optimizer.n_live_points = 20
    phase6.optimizer.sampling_efficiency = 0.8

    # For this phase, we'll also extend it with an inversion phase. This will ensure our pixelization and regularization
    # are fully optimized in conjunction with the hyper-galaxies and background noise-map.

    phase6 = phase6.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=True,
    )

    ### PHASE 7 ###

    # To end, we'll reoptimize the lens light and mass models one final time, using all our hyper-mode features.

    class InversionPhase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_7__lens_sersic_sie__source_inversion"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_6__source_inversion_initialize"
            ).hyper_combined.constant.galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            # Finally, now we trust our source hyper-image, we'll our source-hyper galaxy in this phase.

            if pipeline_settings.hyper_galaxies:

                self.galaxies.lens.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.lens.hyper_galaxy
                )

                self.galaxies.source.hyper_galaxy = (
                    results.last.hyper_combined.constant.galaxies.source.hyper_galaxy
                )

            if pipeline_settings.hyper_image_sky:

                self.hyper_image_sky = (
                    results.last.hyper_combined.constant.hyper_image_sky
                )

            if pipeline_settings.hyper_background_noise:

                self.hyper_background_noise = (
                    results.last.hyper_combined.constant.hyper_background_noise
                )

    phase7 = InversionPhase(
        phase_name="phase_7__lens_sersic_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=0.5,
                light=lp.EllipticalSersic,
                mass=mp.EllipticalIsothermal,
                shear=shear,
            ),
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pipeline_settings.pixelization,
                regularization=pipeline_settings.regularization,
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase7.optimizer.const_efficiency_mode = True
    phase7.optimizer.n_live_points = 75
    phase7.optimizer.sampling_efficiency = 0.2

    phase7 = phase7.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxies,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=True,
    )

    return pipeline.PipelineImaging(
        pipeline_name,
        phase1,
        phase2,
        phase3,
        phase4,
        phase5,
        phase6,
        phase7,
        hyper_mode=True,
    )
