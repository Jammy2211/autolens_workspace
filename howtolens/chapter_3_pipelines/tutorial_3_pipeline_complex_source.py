from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline

def make_pipeline(phase_folders=None):

    pipeline_name = 'pipeline_complex_source'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    # To begin, we need to initialize the lens's mass model. We should be able to do this by using a simple source
    # model. It won't fit the complicated structure of the source, but it'll give us a reasonable estimate of the
    # einstein radius and the other lens-mass parameters.

    # This should run fine without any prior-passes. In general, a thick, giant ring of source light is something we
    # can be confident MultiNest will fit without much issue, especially when the lens galaxy's light isn't included
    # such that the parameter space is just 12 parameters.

    phase1 = ph.LensSourcePlanePhase(phase_name='phase_1_simple_source', phase_folders=phase_folders, 
                                     lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                                     source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalSersic)),
                                     optimizer_class=nl.MultiNest)

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.5

    # Now lets add another source component, using the previous model as the initialization on the lens / source
    # parameters. We'll vary the parameters of the lens mass model and first source galaxy component during the fit.

    class X2SourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_1_simple_source').variable.lens
            self.source_galaxies.source.light_0 = results.from_phase('phase_1_simple_source').variable.source.light_0

    # You'll notice I've stop writing 'phase_1_results = results.from_phase('phase_1_simple_source')' - we know how
    # the previous results are structured now so lets not clutter our code!

    phase2 = X2SourcePhase(phase_name='phase_2_x2_source', phase_folders=phase_folders, 
                           lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalExponential,
                                                                       light_1=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest)

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.5

    # Now lets do the same again, but with 3 source galaxy components.

    class X3SourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_2_x2_source').variable.lens
            self.source_galaxies.source.light_0 = results.from_phase('phase_2_x2_source').variable.source.light_0
            self.source_galaxies.source.light_1 = results.from_phase('phase_2_x2_source').variable.source.light_1

    phase3 = X3SourcePhase(phase_name='phase_3_x3_source', phase_folders=phase_folders, 
                           lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalExponential,
                                                                      light_1=lp.EllipticalSersic,
                                                                      light_2=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest)

    phase3.optimizer.const_efficiency_mode = True
    phase3.optimizer.n_live_points = 50
    phase3.optimizer.sampling_efficiency = 0.5

    # And one more for luck!

    class X4SourcePhase(ph.LensSourcePlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens = results.from_phase('phase_3_x3_source').variable.lens
            self.source_galaxies.source.light_0 = results.from_phase('phase_3_x3_source').variable.source.light_0
            self.source_galaxies.source.light_1 = results.from_phase('phase_3_x3_source').variable.source.light_1
            self.source_galaxies.source.light_2 = results.from_phase('phase_3_x3_source').variable.source.light_2

    phase4 = X4SourcePhase(phase_name='phase_4_x4_source', phase_folders=phase_folders, 
                           lens_galaxies=dict(lens=gm.GalaxyModel(mass=mp.EllipticalIsothermal)),
                           source_galaxies=dict(source=gm.GalaxyModel(light_0=lp.EllipticalExponential,
                                                                      light_1=lp.EllipticalSersic,
                                                                      light_2=lp.EllipticalSersic,
                                                                      light_3=lp.EllipticalSersic)),
                           optimizer_class=nl.MultiNest)

    phase4.optimizer.const_efficiency_mode = True
    phase4.optimizer.n_live_points = 50
    phase4.optimizer.sampling_efficiency = 0.5

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3, phase4)