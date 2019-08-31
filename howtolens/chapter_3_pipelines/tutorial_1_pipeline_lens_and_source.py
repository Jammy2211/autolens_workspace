import autofit as af
import autolens as al
from autolens.pipeline import pipeline

# All pipelines begin with a comment describing the pipeline and a phase-by-phase description of what it does.

# In this pipeline, we'll perform a basic analysis which fits a source galaxy using a parametric light profile and a
# lens galaxy where its light is included and fitted, using three phases:

# Phase 1) Fit the lens galaxy's light using an elliptical Sersic light profile.

# Phase 2) Use this lens subtracted image to fit the lens galaxy's mass (SIE) and source galaxy's light (Sersic).

# Phase 3) Fit the lens's light, mass and source's light simultaneously using priors initialized from the above 2 phases.


def make_pipeline(phase_folders=None):

    # Pipelines takes 'phase_folders' as input, which in conjunction with the pipeline name specify the path structure of
    # the output. In the pipeline runner we pass the phase_folders ['howtolens, c3_t1_lens_and_source], which means the
    # output of this pipeline go to the folder 'autolens_workspace/output/howtolens/c3_t1_lens_and_source/pipeline__light_and_source'.

    # By default, the pipeline folders is None, meaning the output go to the directory 'output/pipeline_name',
    # which in this case would be 'output/pipeline_light_and_source'.

    # In the example pipelines found in 'autolens_workspace/pipelines' folder, we pass the name of our strong lens data
    # to the pipeline path. This allows us to fit a large sample of lenses using one pipeline and store all of their
    # results in an ordered directory structure.

    pipeline_name = "pipeline__light_and_source"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/pipeline_tag/phase_name/phase_tag/'
    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # In chapter 2, we learnt how to mask data. With pipelines we can change our mask between phases. Afterall,
    # the bigger the mask, the slower the run-time. In the early phases of a pipeline we're not bothered about fitting
    # all of the image. Aggresive masking (which removes lots of image-pixels) is an appealing way to get things running fast.

    # In phase 1, we're only interested in fitting the lens's light, so we'll mask out the source-galaxy entirely.
    # This'll give us a nice speed up and ensure the source's light doesn't impact our light-profile fit.

    # We want a mask that is shaped like the source-galaxy. The shape of the source is an 'annulus' (e.g. a ring),
    # so we're going to use an annular mask. For example, if an annulus is specified between an inner radius of 0.5"
    # and outer radius of 2.0", all pixels in two rings between 0.5" and 2.0" are included in the analysis.

    # But wait, we actually want the opposite of this! We want a masks where the pixels between 0.5" and 2.0" are not
    # included! They're the pixels the source is actually located. Therefore, we're going to use an 'anti-annular
    # mask', where the inner and outer radii are the regions we omit from the analysis. This means we need to specify
    # a third mask radii, further out, such that data at these exterior edges of the image are masked.

    # We can set a mask using a 'mask_function', which returns the mask used by a phase.

    def mask_function(image):
        return al.Mask.circular_anti_annular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=1.6,
            outer_radius_2_arcsec=2.5,
        )

    # Next , wecreate the phase, using the same notation we learnt before (noting the masks function is passed to
    # this phase ensuring the anti-annular masks above is used).

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=0.5, light=al.light_profiles.EllipticalSersic)
        ),
        mask_function=mask_function,
        optimizer_class=af.MultiNest,
    )

    # We'll use the MultiNest black magic we covered in tutorial 7 of chapter 2 to get this phase to run fast.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5

    ### PHASE 2 ###

    # In phase 2, we fit the source galaxy's light. Thus, we want to make 2 changes from the previous phase.

    # 1) We want to fit the lens subtracted image calculated in phase 1, instead of the observed image.
    # 2) We want to mask the central regions of this image where there are residuals due to the lens light subtraction.

    # We can use the mask function again, to modify the mask to an annulus. We'll use the same ring radii as before.

    def mask_function(image):
        return al.Mask.circular_annular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.5,
            outer_radius_arcsec=3.0,
        )

    # To modify an image, we call a new function, 'modify image'. This function behaves like the pass-priors functions
    # before, whereby we create a python 'class' in a Phase to set it up.  This ensures it has access to the pipeline's
    # 'results' (which you may have noticed was in the the customize_priors functions as well).

    # To setup the modified image we take the observed image and subtract-off the model image from the
    # previous phase, which, if you're keeping track, is an image of the lens galaxy. However, if we just used the
    # 'model_image' in the fit, this would only include pixels that were masked. We want to subtract the lens off the
    # entire image - fortunately, PyAutoLens automatically generates an 'unmasked_lens_plane_model_image' as well!

    class LensSubtractedPhase(al.PhaseImaging):
        def modify_image(self, image, results):
            phase_1_results = results.from_phase("phase_1__lens_sersic")
            return image - phase_1_results.unmasked_model_image_of_planes[0]

    # The function above demonstrates the most important thing about pipelines - that every phase has access to the
    # results of all previous phases. This means we can feed information through the pipeline and therefore use the
    # results of previous phases to setup new phases.

    # You should see that this is done by using the phase_name of the phase we're interested in, which in the above
    # code is named 'phase_1__lens_sersic' (you can check this on line 73 above).

    # We'll do this again in phase 3 and throughout all of the pipelines in this chapter and the workspace examples.

    # We setup phase 2 as per usual. Note that we don't need to pass the modify image function.

    phase2 = LensSubtractedPhase(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    ### PHASE 3 ###

    # Finally, in phase 3, we want to fit the lens and source simultaneously.

    # We'll use the 'customize_priors' function that we all know and love to do this. However, we're going to use the
    # 'results' argument that, in chapter 2, we ignored. This stores the results of the lens model of
    # phases 1 and 2 meaning we can use it to initialize phase 3's priors!

    class LensSourcePhase(al.PhaseImaging):
        def customize_priors(self, results):

            # The previous results is a 'list' in python. The zeroth index entry of the list maps to the results of
            # phase 1, the first entry to phase 2, and so on.

            phase_1_results = results.from_phase("phase_1__lens_sersic")
            phase_2_results = results.from_phase("phase_2__lens_sie__source_sersic")

            # To link two priors together we invoke the 'variable' attribute of the previous results. By invoking
            # 'variable', this means that:

            # 1) The parameter will be a free-parameter fitted for by the non-linear search.
            # 2) It will use a GaussianPrior based on the previous results as its initialization (we'll cover how this
            #    Gaussian is setup in tutorial 4, for now just imagine it links the results in a sensible way).

            # We can simply link every source galaxy parameter to its phase 2 inferred value, as follows

            self.galaxies.source.light.centre_0 = (
                phase_2_results.variable.galaxies.source.light.centre_0
            )

            self.galaxies.source.light.centre_1 = (
                phase_2_results.variable.galaxies.source.light.centre_1
            )

            self.galaxies.source.light.axis_ratio = (
                phase_2_results.variable.galaxies.source.light.axis_ratio
            )

            self.galaxies.source.light.phi = (
                phase_2_results.variable.galaxies.source.light.phi
            )

            self.galaxies.source.light.intensity = (
                phase_2_results.variable.galaxies.source.light.intensity
            )

            self.galaxies.source.light.effective_radius = (
                phase_2_results.variable.galaxies.source.light.effective_radius
            )

            self.galaxies.source.light.sersic_index = (
                phase_2_results.variable.galaxies.source.light.sersic_index
            )

            # However, listing every parameter like this is ugly and becomes cumbersome if we have a lot of parameters.

            # If, like in the above example, you are making all of the parameters of a lens or source galaxy variable,
            # you can simply set the source galaxy equal to one another without specifying each parameter of every
            # light and mass profile.

            self.galaxies.source = (
                phase_2_results.variable.galaxies.source
            )  # This is identical to lines 196-203 above.

            # For the lens galaxies we have a slightly weird circumstance where the light profiles requires the
            # results of phase 1 and the mass profile the results of phase 2. When passing these as a 'variable', we
            # can split them as follows

            self.galaxies.lens.light = phase_1_results.variable.galaxies.lens.light

            self.galaxies.lens.mass = phase_2_results.variable.galaxies.lens.mass

    phase3 = LensSourcePhase(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=al.light_profiles.EllipticalSersic,
                mass=al.mass_profiles.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(
                redshift=1.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)
