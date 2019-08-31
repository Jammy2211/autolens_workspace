# So, how does hyper-mode work in pipelines? There are a lot more things we have to do now; pass hyper-galaxy-images
# between phases, use different pixelizations and regularization schemes, and decide what parts of the noise-map we do
# and don't want to scale.

### HYPER IMAGE PASSING ###

# So, lets start with hyper-image passing. That is, how do we pass the model images of the lens and source galaxies to
# later phases to use them as hyper-galaxy-images? Well, I've got some good news, *we no nothing*. PyAutoLens automatically
# passes hyper-images between phases!

# However, PyAutoLens does need to know which hyper-images to pass to which galaxies. To do this, PyAutoLens uses
# galaxy-names. When you create a GalaxyModel, you name the galaxies, for example below we've called the lens galaxy
# 'lens' and the source galaxy 'source':

# phase1 = al.PhaseImaging(
#     phase_name="phase_1",
#     phase_folders=phase_folders,
#     galaxies=dict(
# NAME --> lens=al.GalaxyModel(
#             redshift=0.5,
#             light=al.light_profiles.EllipticalSersic,
#             mass=al.mass_profiles.EllipticalIsothermal,
#         )
#     ),
#       galaxies=dict(
# NAME --> source=al.GalaxyModel(redshift=1.0, light=al.light_profiles.EllipticalSersic)
#     ),
# )

# To pass the resulting model images of these galaxies to galaxies in the next phase, galaxies are paired by their names.
# So, in the next phase (phase2), the lens galaxy must be called 'lens' (again) and the source galaxy 'source' (again).
# Hopefully, you've followed this naming convention with PyAutoLens up to now anyway, so this should come naturually.

# Why do we pass images based on galaxy names? It is because for more complex lens systems (e.g. with multiple lens and
# source galaixes) we must explicitly define how their images are linked between phases.

### HYPER PHASES ###

# There is one more major change to hyper-pipelines. That is, how do we fit for the hyper-parameters using our
# non-linear search (e.g. MultiNest)? We don't fit for them simultaneously with the lens and source models, as this
# creates an unnecessarily large paameter space which we'd fail to fit accurately and efficiently.

# Instead, we 'extend' phases with extra phases that specifically fit certain components of hyper-galaxy-model. You've
# hopefully already seen the following code, which optimizes the parameters of an inversion (e.g. the pixelization and
# regularization):

# phase1 = phase1.extend_with_inversion_phase()

# Extending a phase with hyper phases is just as easy:

# phase1 = phase1.extend_with_multiple_hyper_phases(
#     hyper-galaxy=True,
#     include_background_sky=True,
#     include_background_noise=True,
#     inversion=True
# )

# This extends the phase with 3 additional:

# 1) Simultaneously fit the hyper-galaxies, background sky and background noise hyper parameters using the best-fit
#    lens and source models from the main phase. Thus, this phase only scales the noise and the image. This is called
#    the 'hyper-galaxy' phase.

# 2) Fit the inversion parameters using the pixelization and regularization scheme that were used in the main phase.
#    Typically, this fits a brightness-based pixelization and adaptive regularization scheme, but also works for a
#    magnification based pixelization or constant regularization scheme. Again, the previous best-fit lens and source
#    models are used. This is called the 'inversion' phase.

# 3) Fit all of the components above using Gaussian priors centred on the resulting values in steps 1) and 2). This is
#    important as there is trade-off between increasing the noise in the lens / source and changing the pixelization /
#    regularization hyper-galaxy-parameters. This is called the 'hyper_combined' phase.

# In the pipeline below we use the results of the 'hyper_combined' phase to setup the hyper-galaxies, hyper-data,
# pixelizations and regularizations in the next phase. These components are setup as 'constants' and therefore fixed
# during each main phase which fits for the lens and source models.

### HYPER MASKING ###

# In hyper-model, the mask *cannot* change between phases. This is because of hyper-galaxy-image passing. If the mask
# changes and adds new pixels that were not modeled in the previous phase, these pixels wouldn't be in the
# hyper-galaxy-image. This causes some very nasty problems. Thus, the same mask must be used throughout the analysis.
# To ensure this happens, for hyper-galaxy-pipelines we require that the mask is passed to the 'pipeline.run' function.

import autofit as af
import autolens as al

# For our example hyper-pipeline,we're going to run 7 (!) phases. There isn't much new here compared to normal
# pipelines. But, the large number of phases are required to fully model the lens with hyper-mode features. An overview
# of the pipeline is as follows:

# Phase 1) Fit and subtract the lens galaxy's light profile.
# Phase 1 Extension) Fit the lens galaxy's hyper-galaxy and background noise.

# Phase 2) Fit the lens galaxy mass model and source light profile, using the lens subtracted image, using the lens
#          hyper-galaxy noise map and background noise from phase 1.
# Phase 2 Extension) Fit the lens / source hyper-galaxy and background noise.

# Phase 3) Fit the lens light, mass and source using priors from phases 1 & 2, using the lens hyper-galaxy
#          from phase 2.
# Phase 3 Extension) Fit the lens / source hyper-galaxy and background noise.

# Phase 4) Initialize an inversion of the source-galaxy, using a magnification based pixelization and constant
#          regularization scheme.
# Phase 4 Extension) Fit the lens / source hyper-galaxy and background noise.

# Phase 5) Refine the lens light and mass models using the inversion from phase 4.
# Phase 5 Extension) Fit the lens / source hyper-galaxy and background noise.

# Phase 6) Initialize an inversion of the source-galaxy, using a brightness based pixelization and adaptive
#          regularization scheme.
# Phase 6 Extension) Fit the lens / source hyper-galaxy, background noise and reoptimize the inversion.

# Phase 7) Refine the lens light and mass models using the inversion from phase 6 and lens / source hyper galaxy.
# Phase 7 Extension) Fit the lens / source hyper-galaxy, background noise and reoptimize the inversion.

# Phew! Thats a lot of phases, so lets take a look.

# For hyper-pipelines, we pass pipeline settings which detemine whether certain hyper-galaxy features are on or off. For
# example, if pipeline_settings.hyper_galaxy is False, noise-scaling via hyper-galaxies is omitted  We also pass in the
# pixelization / regularization schemes that will be used in phase 6 and 7 of the pipeline.


def make_pipeline(pipeline_settings, phase_folders=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # The pipeline name, tagging and phase folders work exactly like they did in previous phases. However, tagging now
    # also includes the pixelization and regularization schemes, as these cannot be changed for a hyper-galaxy-pipeline.

    pipeline_name = "pipeline__hyper_example"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings(
        pixelization=pipeline_settings.pixelization,
        regularization=pipeline_settings.regularization,
    )

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### PHASE 1 ###

    # We set up and run phase 1 as per usual for pipelines, so nothing new here.

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.light_profiles.EllipticalSersic)
        ),
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    # This extends phase 1 with hyper-phases that fit for the hyper-galaxies, as described above. The extension below
    # adds two phases, a 'hyper-galaxy' phase which fits for the lens hyper-galaxy + the background noise, and a
    # 'hyper_combined' phase which fits them again.
    #
    # Although this might sound like unnecessary repetition, the second phase uses Gaussian priors inferred from the
    # first phase, meaning that it can search regions of parameter space that may of been unaccessible due to the
    # first phase's uniform piors.

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxy,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 2 ###

    # This phase fits for the lens's mass model and source's light profile using the lens subtracted image from phase
    # 1. The lens galaxy hyper-galaxy is included, such that high chi-squared values in the central regions of the
    # image due to a poor lens light subtraction are reduced by noise scaling and do not impact the fit.

    class LensSubtractedPhase(al.PhaseImaging):
        def customize_priors(self, results):

            # Most of this customize priors function is the usual, e.g. passing the lens galaxy's light profile from
            # phase 1 to phase 2.

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

            # These are new to a hyper-galaxy-pipeline and will appear in every pass priors function. Basically, these
            # check whether a hyper-galaxy-feature is turned on. If it is, then it will have been fitted for in the previous
            # phase's 'hyper_combined' phase, so its parameters are passed to this phase as constants.

            if pipeline_settings.hyper_galaxy:

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
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    # This extends phase 2 with hyper-phases that fit for the hyper-galaxies, as described above. This extension again
    # adds two phases, a 'hyper-galaxy' phase and 'hyper_combined' phase. Unlike the extension to phase 1 which only
    # include a lens hyper-galaxy, this extension includes both the lens and source hyper-galaxy galaxies, as well as the
    # background noise.

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxy,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 3 ###

    # Usual stuff in this phase - we fit the lens and source using priors from the above 2 phases.

    class LensSourcePhase(al.PhaseImaging):
        def customize_priors(self, results):
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

            # Although the above hyper-galaxy phase includes fitting for the source galaxy, at this early stage in the
            # pipeline we make a choice not to pass the hyper-galaxy of the source. Why? Because there is a good chance
            # our simplistic single Sersic profile won't yet provide a good fit to the source.
            #
            # If this is the case, the hyper noise map won't be very good. It isn't until we are fitting the
            # source using an inversion that we begin to pass its hyper-galaxy, e.g. when we can be confident our fit
            # to the data is reliable!

            if pipeline_settings.hyper_galaxy:

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
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    # The usual phase extension, which operates the same as the extension for phase 2.

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxy,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
    )

    ### PHASE 4 ###

    # Next, we initialize a magnification based pixelization and constant regularization scheme. But, you're probably
    # wondering, why do we bother with these at all? Why not jump straight to the brightness based pixelization and
    # adaptive regularization?

    # Well, its to do with the hyper-galaxy-images of our source. At the end of phase 3, we've only fitted the source galaxy
    # using a single EllipticalSersic profile. What if the source galaxy is more complex than a Sersic? Or has
    # multiple components? Our fit, put simply, won't be very good! This makes for a bad hyper-galaxy-image.

    # So, its beneficial for us to introduce an intermediate inversion using a magnification based grid, that fits
    # all components of the source accurately giving us a good quality hyper-galaxy image for the brightness based
    # pixelization and adaptive regularization. Its for this reason we've also omitted the hyper-galaxy source galaxy
    # from the phases above; if the hyper-galaxy-image were poor, so is the hyper noise-map!

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).constant.galaxies.lens

            ## Set all hyper-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxy:

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
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pixelizations.VoronoiMagnification,
                regularization=al.regularization.Constant,
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    # This is the usual phase extensions. Given we're only using this inversion to refine our hyper-galaxy-images, we
    # won't bother reoptimizing its hyper-galaxy-parameters

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxy,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 5 ###

    # Here, we refine the lens light and mass models using this magnification based pixelization and constant
    # regularization scheme. If our source model was a bit dodgy because it was more complex than a single Sersic, it
    # probably means our lens model was too, so a quick refinement in phase 5 is worth the effort!

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_3__lens_sersic_sie__source_sersic"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_4__source_inversion_initialize_magnification"
            ).constant.galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxy:

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
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
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
        hyper_galaxy=pipeline_settings.hyper_galaxy,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 6 ###

    # In phase 6, we finally use our hyper-galaxy-mode features to adapt the pixelization to the source's morphology
    # and the regularization to its brightness! This phase works like a hyper initialization phase, whereby we fix
    # the lens and source models to the best-fit of the previous phase and just optimize the hyper-galaxy-parameters.

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_5__lens_sersic_sie__source_inversion_magnification"
            ).constant.galaxies.lens

            ## Set all hyper-galaxies if feature is turned on ##

            if pipeline_settings.hyper_galaxy:

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
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
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

    # For this phase, we'll also extend it with an inversion phase. This ensures our pixelization and regularization
    # are fully optimized in conjunction with the hyper-galaxies and background noise-map.

    phase6 = phase6.extend_with_multiple_hyper_phases(
        hyper_galaxy=pipeline_settings.hyper_galaxy,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=True,
    )

    ### PHASE 7 ###

    # To end, we reoptimize the lens light and mass models one final time using all our hyper-galaxy-mode features.

    class InversionPhase(al.PhaseImaging):
        def customize_priors(self, results):
            ## Lens Light & Mass, Sersic -> Sersic, SIE -> SIE, Shear -> Shear ###

            self.galaxies.lens = results.from_phase(
                "phase_7__lens_sersic_sie__source_inversion"
            ).variable.galaxies.lens

            ### Source Inversion, Inv -> Inv ###

            self.galaxies.source = results.from_phase(
                "phase_6__source_inversion_initialize"
            ).hyper_combined.constant.galaxies.source

            ## Set all hyper-galaxies if feature is turned on ##

            # Finally, now we trust our source hyper-galaxy-image, we'll our source-hyper-galaxy in this phase.

            if pipeline_settings.hyper_galaxy:

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
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
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
        hyper_galaxy=pipeline_settings.hyper_galaxy,
        include_background_sky=pipeline_settings.hyper_image_sky,
        include_background_noise=pipeline_settings.hyper_background_noise,
        inversion=True,
    )

    return al.PipelineImaging(
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
