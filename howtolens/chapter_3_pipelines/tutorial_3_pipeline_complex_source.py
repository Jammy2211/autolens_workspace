import autofit as af
import autolens as al
from autolens.pipeline import pipeline

# Up to now, we have passed the priors between phases using the 'customize priors' function. This works nicely, and
# gives us a lot of control for how the prior on every individual parameter is specified. However, it also makes
# the pipeline code longer than it needs to be, and it does not check for typos or errors in the prior linking.

# In this pipeline, we'll pass priors in a slight different way, where the results of a phase are directly passed to
# the next phase. This does not change the behaviour of the pipeline from the previous pipelines, but as you'll see
# reduces the amount of code.

# All template pipelines found in the autolens_workspace use this method of prior passing, so its worth you learning!


def make_pipeline(phase_folders=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below. However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline__complex_source"

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/pipeline_tag/phase_name/phase_tag/'
    phase_folders.append(pipeline_name)

    # To begin, we need to initialize the lens's mass model. We should be able to do this by using a simple source
    # model. It won't fit the complicated structure of the source, but it'll give us a reasonable estimate of the
    # einstein radius and the other lens-mass parameters.

    # This should run fine without any prior-passes. In general, a thick, giant ring of source light is something we
    # can be confident MultiNest will fit without much issue, especially when the lens galaxy's light isn't included
    # such that the parameter space is just 12 parameters.

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=al.mass_profiles.EllipticalIsothermal
            ),
            source=al.GalaxyModel(
                redshift=1.0, light_0=al.light_profiles.EllipticalSersic
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.5

    # Now lets add another source component, using the previous model as the initialization on the lens / source
    # parameters. We'll vary the parameters of the lens mass model and first source galaxy component during the fit.

    # To set up phase 2, we use the new method of passing priors, by directly passing the results of phase 1 to the
    # appropirate model components. The 'variable' behaves exactly as it did in the 'customize_priors' function. Hopefully
    # you'll agree the code below is a lot more concise than using the customize_priors functioon!

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_sie__source_x2_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase1.result.variable.galaxies.lens
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                light_0=phase1.result.variable.galaxies.source.light_0,
                light_1=al.light_profiles.EllipticalSersic,
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.5

    # Now lets do the same again, but with 3 source galaxy components.

    phase3 = al.PhaseImaging(
        phase_name="phase_3__lens_sie__source_x3_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase2.result.variable.galaxies.lens
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                light_0=phase2.result.variable.galaxies.source.light_0,
                light_1=phase2.result.variable.galaxies.source.light_1,
                light_2=al.light_profiles.EllipticalSersic,
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    # And one more for luck!

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_sie__source_x4_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=0.5, mass=phase3.result.variable.galaxies.lens
            ),
            source=al.GalaxyModel(
                redshift=1.0,
                light_0=phase3.result.variable.galaxies.source.light_0,
                light_1=phase3.result.variable.galaxies.source.light_1,
                light_2=phase3.result.variable.galaxies.source.light_2,
                light_3=al.light_profiles.EllipticalSersic,
            ),
        ),
        optimizer_class=af.MultiNest,
    )

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4)
