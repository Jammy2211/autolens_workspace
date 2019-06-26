import autofit as af
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging, phase_hyper
from autolens.pipeline import pipeline
from autolens.pipeline import tagging as tag
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.profiles import light_and_mass_profiles as lmp
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

# In this pipeline, we'll perform a basic analysis which initializes a lens model (the lens's light, mass and source's \
# light) and then fits the source galaxy using an inversion. This pipeline uses three phases:

# Phase 1:

# Description: Initializes and subtracts the lens light model.
# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: None
# Source Light: None
# Previous Pipelines: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Description: Initializes the lens mass model and source light profile, using the lens subtracted image from phase 1.
# Lens Light: None
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Uses the lens light subtracted image from phase 1

# Phase 3:

# Description: Refine the lens light and mass models and source light profile, using priors from the previous 2 phases.
# Lens Light: EllipticalSersic + EllipticalExponential
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens light (variable -> phase 1), lens mass and source (variable -> phase 2)
# Notes: None

def make_pipeline(phase_folders=None, tag_phases=True, redshift_lens=0.5, redshift_source=1.0):

    pipeline_name = 'example_pipeline'

    ### PHASE 1 ###

    phase1 = phase_imaging.LensPlanePhase(
        phase_name='phase_1_lens_sersic_exp', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge=lp.EllipticalSersic,
                disk=lp.EllipticalExponential)),
        optimizer_class=af.MultiNest)

    ### PHASE 2 ###

    class LensSubtractedPhase(phase_imaging.LensSourcePlanePhase):

        def modify_image(self, image, results):
            return image - results.from_phase('phase_1_lens_sersic_exp').best_fit.model_image_2d

    phase2 = LensSubtractedPhase(
        phase_name='phase_2_lens_sie_shear_source_sersic', phase_folders=phase_folders, tag_phases=tag_phases,
        lens_galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                mass=mp.EllipticalIsothermal)),
        source_galaxies=dict(
            source=gm.GalaxyModel(
                redshift=redshift_source,
                light=lp.EllipticalSersic)),
        optimizer_class=af.MultiNest)


    phase3 = phase_imaging.MultiPlanePhase(
        phase_name='phase_3_multi_plane', phase_folders=phase_folders, tag_phases=tag_phases,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens,
                bulge=lmp.EllipticalSersic,
                disk=lmp.EllipticalExponential,
                dark=mp.EllipticalGeneralizedNFW,
                shear=mp.ExternalShear),
            los_0=gm.GalaxyModel(
                redshift=0.2,
                mass=mp.SphericalIsothermal),
            los_1=gm.GalaxyModel(
                redshift=0.8,
                mass=mp.SphericalIsothermal),
            los_2=gm.GalaxyModel(
                variable_redshift=True,
                mass=mp.SphericalIsothermal),
            source=gm.GalaxyModel(
                redshift=redshift_source,
                pixelization=pix.VoronoiMagnification,
                regularization=reg.Constant)),
        optimizer_class=af.MultiNest)

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2, phase3)