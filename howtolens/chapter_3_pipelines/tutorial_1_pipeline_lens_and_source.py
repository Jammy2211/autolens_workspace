import autofit as af
import autolens as al

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

    # In the example pipelines found in 'autolens_workspace/pipelines' folder, we pass the name of our strong lens dataset
    # to the pipeline path. This allows us to fit a large sample of lenses using one pipeline and store all of their
    # results in an ordered directory structure.

    pipeline_name = "pipeline__light_and_source"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/pipeline_tag/phase_name/phase_tag/'
    phase_folders.append(pipeline_name)

    ### PHASE 1 ###

    # First, we create the phase, using the same notation we learnt before (noting the masks function is passed to
    # this phase ensuring the anti-annular masks above is used).

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=al.GalaxyModel(redshift=0.5, light=al.lp.EllipticalSersic)),
        optimizer_class=af.MultiNest,
    )

    # We'll use the MultiNest black magic we covered in tutorial 7 of chapter 2 to get this phase to run fast.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.5
    phase1.optimizer.evidence_tolerance = 100.0

    ### PHASE 2 ###

    # In phase 2, we fit the source galaxy's light. Thus, we want to fix the lens light model to the model inferred
    # in phase 1, ensuring the image we fit is lens subtracted. We do this below by passing the lens light as an
    # 'instance' object, a trick we'll use again in the next pipeline in this chapter.

    # To be clear, when we pass an 'instance', we are telling PyAutoLens that we want it to pass the best-fit result of
    # that phase and use those parameters as fixed values in the model. Thus, phase2 below essentially includes a lens
    # light model that is used every time the model fit is performed, however the parameters of the lens light are no
    # longer free parameters but instead fixed values.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5,
                light=phase1.result.instance.galaxies.lens.light,
                mass=al.mp.EllipticalIsothermal,
            ),
            source=al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic),
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3
    phase2.optimizer.evidence_tolerance = 100.0

    ### PHASE 3 ###

    # Finally, in phase 3, we want to fit the lens and source simultaneously. First, lets set up our lens as a
    # GalaxyModel.

    lens = al.GalaxyModel(
        redshift=0.5, light=al.lp.EllipticalSersic, mass=al.mp.EllipticalIsothermal
    )
    source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

    # To link two priors together we invoke the 'model' attribute of the previous results. By invoking
    # 'model', this means that:

    # 1) The parameter will be a free-parameter fitted for by the non-linear search.
    # 2) It will use a GaussianPrior based on the previous results as its initialization (we'll cover how this
    #    Gaussian is setup in tutorial 4, for now just imagine it links the results in a sensible way).

    # We can simply link every source galaxy parameter to its phase 2 inferred value, as follows

    source.light.centre_0 = phase2.result.model.galaxies.source.light.centre_0

    source.light.centre_1 = phase2.result.model.galaxies.source.light.centre_1

    source.light.axis_ratio = phase2.result.model.galaxies.source.light.axis_ratio

    source.light.phi = phase2.result.model.galaxies.source.light.phi

    source.light.intensity = phase2.result.model.galaxies.source.light.intensity

    source.light.effective_radius = (
        phase2.result.model.galaxies.source.light.effective_radius
    )

    source.light.sersic_index = phase2.result.model.galaxies.source.light.sersic_index

    # However, listing every parameter like this is ugly and becomes cumbersome if we have a lot of parameters.

    # If, like in the above example, you are making all of the parameters of a lens or source galaxy variable,
    # you can simply set the source galaxy equal to one another without specifying each parameter of every
    # light and mass profile.

    source = (
        phase2.result.model.galaxies.source
    )  # This is identical to lines 196-203 above.

    # For the lens galaxies we have a slightly weird circumstance where the light profiles requires the
    # results of phase 1 and the mass profile the results of phase 2. When passing these as a 'model', we
    # can split them as follows

    lens.light = phase1.result.model.galaxies.lens.light
    lens.mass = phase2.result.model.galaxies.lens.mass

    # Passing results as a 'model' contrasts our use of an 'instance' in phase2 above, when we passed the lens light
    # parameters as fixed value that were not fitted. In summary:

    # - model: This means we pass the best-fit parameters of a phase, and set them up in the next phase as free
    #          parameters that are fitted for by MultiNest.

    # - instance: This means we pass the best-fit parameters of a phase as fixed parameters that are not fitted for.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sersic_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(lens=lens, source=source),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
