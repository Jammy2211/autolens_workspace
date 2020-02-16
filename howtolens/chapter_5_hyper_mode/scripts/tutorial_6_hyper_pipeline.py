# so, you're probably wondering, how does hyper_galaxies-mode work in the context of pipelines? Afterall, there are a lot more
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
# NAME --> lens=al.GalaxyModel(
#             redshift=0.5,
#             light=al.lp.EllipticalSersic,
#             mass=al.mp.EllipticalIsothermal,
#         )
#     ),
#       galaxies=dict(
# NAME --> source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)
#     ),
# )

# To pass the resulting model images of these galaxies to galaxies in the next phase galaxies are paired by their names.
# So, the lens galaxy would need to be called 'lens' again, and the source galaxy 'source' again. This is probably the
# naming convention you've followed using PyAutoLens up to now anyway, so this should come naturually.
#
# The reason we use this galaxy naming scheme is for more complex lens systems, which may have multiple lens and source
# galaixes. In these cases, it is important that we can link galaxies together by explicitly, using their names.

### HYPER PHASES ###

# There is only one more big change to hyper_galaxies-pipelines from pipelines. That is, how do we fit for the
# hyper_galaxies-parameters using our non-linear search (e.g. MultiNest)? What we don't want to do is fit for them all whilst
# also fitting the lens and source models. That would create an unnecessarily large paameter space, which we'd fail to
# fit accurately and efficiently.

# So, we instead 'extend' phases with extra phases that specifically fit certain components of hyper_galaxies-mode. You've
# hopefully already seen this with the following line of code, which optimizes the hyper_galaxies-parameters of just the
# inverison (e.g. the pixelization and regularization):

# phase1 = phase1.extend_with_inversion_phase()

# Extending a phase with hyper_galaxies phases is just as easy:

# phase1 = phase1.extend_with_multiple_hyper_phases(
#     hyper_galaxies=True,
#     include_background_sky=True,
#     include_background_noise=True,
#     inversion=True
# )

# This extends te phase with 3 addition phases, which do the following:

# 1) Simultaneously fit all hyper-galaxies, the background sky and background noise hyper_galaxies parameters, using the best-fit
#    lens and source models from the hyper phase. Thus, this phase only scales the noise and the image. This is called
#    the 'hyper_galaxies' phase.

# 2) Fit all of the inversion parameters, using the pixelization and regularization scheme that were used in the hyper
#    phase. Typically, this would be a brightness-based pixelization and adaptive regularization scheme, but if it were
#    magnification based and constant it would work just fine. Again, the previous best-fit lens and source models are
#    used. This is called the 'inversion' phase.

# 3) Fit all of the components above, using Gaussian priors centred on the resulting values in steps 1) and 2). This is
#    important as there is trade-off between increasing the noise in the lens / source and changing the pixelization /
#    regularization hyper_galaxies-parameters. This is called the 'hyper_combined' phase.

# In the pipeline below, you'll see we use the results of these phases (typically just the 'hyper_combined' phase) to
# setup the hyper-galaxies, hyper_galaxies-data, pixelization and regularization in the next phase. Infact, in this pipeline
# all these components will then be setup as 'constants' and therefore fixed during each phases 'hyper' optimization
# that changes lens and source models.

### HYPER MASKING ###

# There is one more rule you should know for hyper_galaxies-mode. That is, the mask *cannot* change between phases. This is
# because of hyper-image passing. Basically, if you were to change a mask such that it adds new, extra pixels that
# were not modeled in the previous phase, these pixels wouldn't be contained in the hyper-image. This causes some
# very nasty problems. So, instead, we simply ask that you use the same mask throughout the analysis. To ensure
# this happens, for hyper_galaxies-pipelines we require that the mask is passed to the 'pipeline.run' function.

import autofit as af
import autolens as al
import autolens.plot as aplt

# Okay, so for our example hyper_galaxies-pipeline, we're going to run 7 (!) phases. Conceptually, there isn't too much new
# here compared to pipelines. But, we need so many phases to explain a few design choices that arise for
# hyper_galaxies pipelines. An overview of the pipeline is as follows:

# Phase 1) Fit and subtract the lens galaxies light profile.
# Phase 1 Extension) Fit the lens galaxy hyper_galaxies galaxy and background noise.

# Phase 2) Fit the lens galaxy mass model and source light profile, using the lens subtracted image, the lens
#          hyper_galaxies-galaxy noise map and background noise from phase 1.
# Phase 2 Extension) Fit the lens / source hyper_galaxies galaxies and background noise.

# Phase 3) Fit the lens light, mass and source using priors from phases 1 & 2, and using the hyper_galaxies lens / source
#          galaxies from phase 2.
# Phase 3 Extension) Fit the lens / source hyper_galaxies galaxies and background noise.

# Phase 4) Initialize an inversion of the source-galaxy, using a magnification based pixelization and constant
#          regularization scheme.
# Phase 4 Extension) Fit the lens / source hyper_galaxies galaxies and background noise)

# Phase 5) Refine the lens light and mass models using the inversion from phase 4.
# Phase 5 Extension) Fit the lens / source hyper_galaxies galaxies and background noise.

# Phase 6) Initialize an inversion of the source-galaxy, using a brightness based pixelization and adaptive
#          regularization scheme.
# Phase 6 Extension) Fit the lens / source hyper_galaxies galaxies, background noise and reoptimize the inversion.

# Phase 7) Refine the lens light and mass models using the inversion from phase 6.
# Phase 7 Extension) Fit the lens / source hyper_galaxies galaxies, background noise and reoptimize the inversion.

# Phew! Thats a lot of phases, so lets take a look.

# For hyper_galaxies-pipelines, we pass bools into the function to determine whether certain hyper_galaxies features are on or off. For
# example, if pipeline_setup.hyper_galaxies is False, then noise-scaling via hyper-galaxies will be omitted from this run of the
# pipeline. We also pass in the pixelization / regularization schemes that will be used in phase 6 and 7 of the
# pipeline.


def make_pipeline(general_setup, source_setup, phase_folders=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The pixelization and regularization scheme of the pipeline (fitted in phases 4 & 5).

    pipeline_name = "pipeline__hyper_example"

    phase_folders.append(pipeline_name)
    phase_folders.append(general_setup.tag + source_setup.tag)

    ### PHASE 1 ###

    # We set up and run phase 1 as per usual for pipelines, so nothing new here.

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)),
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    # This extends phase 1 with hyper_galaxies-phases that fit for the hyper-galaxies, as described above. The extension below
    # adds two phases, a 'hyper_galaxies-galaxy' phase which fits for the lens hyper_galaxies galaxy + the background noise, and a
    # 'hyper_combined' phase which fits them again.
    #
    # Although this might sound like unnecessary repetition, the second phase uses Gaussian priors inferred from the
    # first phase, meaning that it can search regions of parameter space that may of been unaccessible due to the
    # first phase's uniform piors.

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy=general_setup.hyper_galaxies,
        include_background_sky=general_setup.hyper_image_sky,
        include_background_noise=general_setup.hyper_background_noise,
    )

    ### PHASE 2 ###

    # This phase fits for the lens's mass model and source's light profile, using the lens subtracted image from phase
    # 1. The lens galaxy hyper_galaxies-galaxy will be included, such that high chi-squared values in the central regions of the
    # image due to a poor lens light subtraction are reduced by noise scaling and do not significantly impact the fit.

    class LensSubtractedPhase(al.PhaseImaging):
        def customize_priors(self, results):

            ## Lens Mass, Move centre priors to centre of lens light ###

            self.galaxies.lens.mass.centre = (
                results.from_phase("phase_1__lens_sersic")
                .model_absolute(a=0.1)
                .galaxies.lens.light.centre
            )

    # The hyper_galaxy, hyper_image_sky and hyper_background_noise are new to a hyper_galaxies-pipeline, and will appear
    # in every pass priors function. Basically, these check whether a hyper_galaxies-feature is turned on. If it is,
    # then it will have been fitted for in the previous phase's 'hyper_combined' phase, so its parameters are passed to
    # this phase as constants.

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.instance.galaxies.lens.light,
                mass=al.mp.EllipticalIsothermal,
                shear=al.mp.ExternalShear,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    # This extends phase 2 with hyper_galaxies-phases that fit for the hyper-galaxies, as described above. This extension again
    # adds two phases, a 'hyper_galaxies' phase and 'hyper_combined' phase. Unlike the extension to phase 1 which only
    # include a lens hyper_galaxies-galaxy, this extension includes both the lens and source hyper_galaxies galaxies, as well as the
    # background noise.

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=general_setup.hyper_galaxies,
        include_background_sky=general_setup.hyper_image_sky,
        include_background_noise=general_setup.hyper_background_noise,
    )

    ### PHASE 3 ###

    # Usual stuff in this phase - we fit the lens and source using priors from the above 2 phases.

    ## Set all hyper-galaxies if feature is turned on ##

    # Although the above hyper_galaxies phase includes fitting for the source galaxy, at this early stage in the pipeline
    # we make a choice not to pass the hyper_galaxies-galaxy of the source. Why? Because there is a good chance
    # our simplistic single Sersic profile won't yet provide a good fit to the source.
    #
    # If this is the case, it hyper noise map won't be very good. It isn't until we are fitting the
    # source using an inversion that we begin to pass its hyper_galaxies galaxy, e.g. when we can be confident our fit
    # to the dataset is reliable!

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.model.galaxies.lens.light,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
                hyper_galaxy=phase2.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(light=phase2.result.model.galaxies.source.light),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 75
    phase3.optimizer.sampling_efficiency = 0.3

    # The usual phase extension, which operates the same as the extension for phase 2.

    phase3 = phase3.extend_with_multiple_hyper_phases(
        hyper_galaxy=general_setup.hyper_galaxies,
        include_background_sky=general_setup.hyper_image_sky,
        include_background_noise=general_setup.hyper_background_noise,
    )

    ### PHASE 4 ###

    # Next, we initialize a magnification based pixelization and constant regularization scheme. But, you're probably
    # wondering, why do we bother with these at all? Why not jump straight to the brightness based pixelization and
    # adaptive regularization?

    # Well, its to do with the hyper-images of our source. At the end of phase 3, we've only fitted the source galaxy
    # using a single EllipticalSersic profile. What if the source galaxy is more complex than a Sersic? Or has
    # multiple components? Our fit, put simply, won't be very good! This makes for a bad hyper-image.

    # So, its beneficial for us to introduce an intermediate inversion using a magnification based grid, that will fit
    # all components of the source accurately, so that we have a good quality hyper_galaxies image for the brightness based
    # pixelization and adaptive regularization. Its for this reason we've also omitted the hyper_galaxies source galaxy from
    # the phases above; if the hyper-image were poor, so is the hyper noise-map!

    phase4 = al.PhaseImaging(
        phase_name="phase_4__source_inversion_initialize_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase3.result.instance.galaxies.lens.light,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase3.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 20
    phase4.optimizer.sampling_efficiency = 0.8

    # This is the usual phase extensions. Given we're only using this inversion to refine our hyper-images, we won't
    # bother reoptimizing its hyper_galaxies-parameters

    phase4 = phase4.extend_with_multiple_hyper_phases(
        hyper_galaxy=general_setup.hyper_galaxies,
        include_background_sky=general_setup.hyper_image_sky,
        include_background_noise=general_setup.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 5 ###

    # Here, we refine the lens light and mass models using this magnification based pixelization and constant
    # regularization scheme. If our source model was a bit dodgy because it was more complex than a single Sersic, it
    # probably means our lens model was too, so a quick refinement in phase 5 is worth the effort!

    phase5 = al.PhaseImaging(
        phase_name="phase_5__lens_sersic_sie__source_inversion_magnification",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase3.result.model.galaxies.lens.light,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
                hyper_galaxy=phase4.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
            ),
        ),
        hyper_image_sky=phase4.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase4.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase5.optimizer.const_efficiency_mode = True
    phase5.optimizer.n_live_points = 75
    phase5.optimizer.sampling_efficiency = 0.2

    phase5 = phase5.extend_with_multiple_hyper_phases(
        hyper_galaxy=general_setup.hyper_galaxies,
        include_background_sky=general_setup.hyper_image_sky,
        include_background_noise=general_setup.hyper_background_noise,
        inversion=False,
    )

    ### PHASE 6 ###

    # In phase 6, we finally use our hyper_galaxies-mode features to adapt the pixelization to the source's morphology
    # and the regularization to its brightness! This phase works like a hyper initialization phase, whereby we fix
    # the lens and source models to the best-fit of the previous phase and just optimize the hyper_galaxies-parameters.

    phase6 = al.PhaseImaging(
        phase_name="phase_6__source_inversion_initialize",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase5.result.instance.galaxies.lens.light,
                mass=phase5.result.instance.galaxies.lens.mass,
                shear=phase5.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase5.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=source_setup.pixelization,  # <- This is our brightness based pixelization provided it was input into the pipeline.
                regularization=source_setup.regularization,  # <- And this our adaptive regularization.
            ),
        ),
        hyper_image_sky=phase5.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase5.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase6.optimizer.const_efficiency_mode = True
    phase6.optimizer.n_live_points = 20
    phase6.optimizer.sampling_efficiency = 0.8

    # For this phase, we'll also extend it with an inversion phase. This will ensure our pixelization and regularization
    # are fully optimized in conjunction with the hyper-galaxies and background noise-map.

    phase6 = phase6.extend_with_multiple_hyper_phases(
        hyper_galaxy=general_setup.hyper_galaxies,
        include_background_sky=general_setup.hyper_image_sky,
        include_background_noise=general_setup.hyper_background_noise,
        inversion=True,
    )

    ### PHASE 7 ###

    # To end, we'll reoptimize the lens light and mass models one final time, using all our hyper_galaxies-mode features.

    # Finally, now we trust our source hyper-image, we'll our source-hyper_galaxies galaxy in this phase.

    phase7 = al.PhaseImaging(
        phase_name="phase_7__lens_sersic_sie__source_inversion",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase5.result.model.galaxies.lens.light,
                mass=phase5.result.model.galaxies.lens.mass,
                shear=phase5.result.model.galaxies.lens.shear,
                hyper_galaxy=phase6.result.hyper_combined.instance.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                pixelization=phase6.result.instance.galaxies.source.pixelization,
                regularization=phase6.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase6.result.hyper_combined.instance.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase6.result.hyper_combined.instance.hyper_image_sky,
        hyper_background_noise=phase6.result.hyper_combined.instance.hyper_background_noise,
        optimizer_class=af.MultiNest,
    )

    phase7.optimizer.const_efficiency_mode = True
    phase7.optimizer.n_live_points = 75
    phase7.optimizer.sampling_efficiency = 0.2

    phase7 = phase7.extend_with_multiple_hyper_phases(
        hyper_galaxy=general_setup.hyper_galaxies,
        include_background_sky=general_setup.hyper_image_sky,
        include_background_noise=general_setup.hyper_background_noise,
        inversion=True,
    )

    return al.PipelineDataset(
        pipeline_name, phase1, phase2, phase3, phase4, phase5, phase6, phase7
    )
