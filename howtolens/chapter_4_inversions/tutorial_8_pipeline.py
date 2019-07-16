import autofit as af
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy_model as gm
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg
from autolens.pipeline.phase import phase_imaging, phase_extensions
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag

# In this pipeline, we'll perform a basic analysis which fits a source galaxy using an inversion and a
# lens galaxy where its light is not included and fitted, using two phases:

# Phase 1) Fit the lens galaxy's mass (SIE) and source galaxy's light (Sersic).

# Phase 2) Fit the lens galaxy's mass (SIE) and source galaxy's light using an inversion, where the SIE mass model
#          priors are initialized from phase 1. This new SIE mass model will be used to refine the inversion's
#          pixelization and regularization parameters.

# Phase 3) Fit the lens galaxy's mass (SIE) and source galaxy's light using another inversion, which has had the
#           pixelization and regularization refined from phase 2.


def make_pipeline(phase_folders=None):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pl__inversion"

    pipeline_name = tag.pipeline_name_from_name_and_settings(
        pipeline_name=pipeline_name
    )

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders.append(pipeline_name)

    # This is the same phase 1 as the complex source pipeline, which we saw gave a good fit to the overall
    # structure of the lensed source and provided an accurate lens mass model.

    phase1 = phase_imaging.LensSourcePlanePhase(
        phase_name="phase_1_initialize",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(redshift=1.0, light=lp.EllipticalSersic)
        ),
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.sampling_efficiency = 0.3
    phase1.optimizer.const_efficiency_mode = True

    # Now, in phase 2, lets use the lens mass model to fit the source with an inversion.

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):

            # We can customize the inversion's priors like we do our light and mass profiles.

            self.lens_galaxies.lens = results.from_phase(
                "phase_1_initialize"
            ).variable.lens_galaxies.lens

            self.source_galaxies.source.pixelization.shape_0 = af.UniformPrior(
                lower_limit=20.0, upper_limit=40.0
            )

            self.source_galaxies.source.pixelization.shape_1 = af.UniformPrior(
                lower_limit=20.0, upper_limit=40.0
            )

            # The expected value of the regularization coefficient depends on the details of the data reduction and
            # source galaxy. A broad log-uniform prior is thus an appropriate way to sample the large range of
            # possible values.

            self.source_galaxies.source.regularization.coefficient = af.LogUniformPrior(
                lower_limit=1.0e-6, upper_limit=10000.0
            )

    phase2 = InversionPhase(
        phase_name="phase_2_inversion_initial",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            )
        ),
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.sampling_efficiency = 0.3
    phase2.optimizer.const_efficiency_mode = True

    # This function 'extends' phase 2, with an additional 'inversion phase'. An inversion phase is a standard phase
    # which automatically uses the best-fit model of the phase it is extending (in this example, phase 2) to refine the
    # inversion by optimizing only the pixelization & regularization parameters.

    # The results of this inversion phase are accessible in their own attribute of the results, called 'inversion, as
    # we'll see in phase 3.

    phase2 = phase2.extend_with_inversion_phase()

    # Now, in phase 3, lets use the refined source inversion to fit the lens mass model again.

    class InversionPhase(phase_imaging.LensSourcePlanePhase):
        def pass_priors(self, results):
            # We can customize the inversion's priors like we do our light and mass profiles.

            self.lens_galaxies.lens = results.from_phase(
                "phase_2_inversion_initial"
            ).variable.lens_galaxies.lens

            # Note the use of the 'inversion' result attribute below, which uses the extended inversion phase results.

            self.source_galaxies.source = results.from_phase(
                "phase_2_inversion_initial"
            ).inversion.constant.source_galaxies.source

    phase3 = InversionPhase(
        phase_name="phase_3_inversion_final",
        phase_folders=phase_folders,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(redshift=0.5, mass=mp.EllipticalIsothermal)
        ),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=1.0,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant,
            )
        ),
        optimizer_class=af.MultiNest,
    )

    phase3.optimizer.sampling_efficiency = 0.3
    phase3.optimizer.const_efficiency_mode = True

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)
