from autofit.tools import path_util
from autofit.optimize import non_linear as nl
from autofit.mapper import prior
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline import phase as ph
from autolens.pipeline import pipeline
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

# This pipelines assumes a continuation from the following initialization pipelines:

# -'pipelines/initializers/lens_sersic_sie_source_sersic_from_init.py'
# -'pipelines/initializers/lens_sersic_sie_source_inversion_from_pl.py'

# It links to these pipeline as follows:

# - The Sersic light profile initializes the Sersic + Exponential parameters fitted in this pipeline.
# - The SIE mass profile initializes the SIE mass profiles fitted in this phase.
# - The source inversion initializes the source inversion fitted in this phase.

# In this pipeline, we'll fit the lens galaxy with a Sersic (bulge) + Expoentnial (envelope) model, first assuming a
# fullly aligned geometry (e.g. the bulge and envelope share the same centre, orientatioon), then assuming that they
# don't. This pipeline uses two phases:

# Phase 1) Fit the lens galaxy's light using a Sersic (bulge) + Exponential (Exp) light model, where their geometric
#          parameters are assumed to be the same. The priors are inialized usinig the lens light profile from the
#          previous pipeline. The lens mass and source inversion also use the previous pipelines results.

# Phase 2) Fit the lens galaxy's light using the same Sersic + Exponential model, but not allowing their centres and
#          rotation angle phi to vary relative to one another.

def make_pipeline(phase_folders=None):

    pipeline_name = 'pl_main_lens_sersic_exp_sie_source_inversion'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/phase_folder_1/phase_folder_2/pipeline_name/phase_name/'
    phase_folders = path_util.phase_folders_from_phase_folders_and_pipeline_name(phase_folders=phase_folders,
                                                                                pipeline_name=pipeline_name)

    ### PHASE 1 ###

    # In phase 1, we will fit only the lens galaxy's light, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    # 2) Use a circular mask which includes the lens and source galaxy light.

    class LensPhase(ph.LensPlanePhase):

        def pass_priors(self, results):

            self.lens_galaxies.lens.light.centre_0 = prior.GaussianPrior(mean=0.0, sigma=0.1)
            self.lens_galaxies.lens.light.centre_1 = prior.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensPhase(phase_name='phase_1_lens_light_only', phase_folders=phase_folders,
                       lens_galaxies=dict(lens=gm.GalaxyModel(light=lp.EllipticalSersic)),
                       optimizer_class=nl.MultiNest, mask_function=mask_function_circular)

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout the tutorial '' in howtolens/chapter_2).

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 30
    phase1.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)