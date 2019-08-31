import autofit as af
import autolens as al


# In this pipeline, we'll demonstrate passing redshifts to a pipeline - which means that the results and images of this
# pipeline will be returned in physical units (e.g. lengths in kpcs as well as arcsec, luminosities in magnitudes,
# masses in solMass, etc).

# The redshift of the lens and source are input parameters of all pipelines, and they take default values of 0.5 and
# 1.0. Thus, *all* pipelines will return physical values assuming these fiducial values if no other values are
# specified. Care must be taken interpreting the distances and masses if these redshifts are not correct or if the
# true redshifts of the lens and / or source galaxies are unknown.

# We'll perform a basic analysis which fits a lensed source galaxy using a parametric light profile where
# the lens's light is omitted. This pipeline uses two phases:

# Phase 1:

# Description: Fit the lens mass model and source light profile using x1 source.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: Inputs the pipeline default redshifts where the lens has redshift 0.5, source 1.0.

# Phase 1:

# Description: Fits the lens and source model again..
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), source light (variable -> phase 1)
# Notes: Manually over-rides the lens redshift to 1.0 and source redshift to 2.0, to illustrate the different results.


def make_pipeline(phase_folders=None, redshift_lens=0.5, redshift_source=1.0):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below. However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_feature__redshifts_lens__source"

    pipeline_tag = al.pipeline_tagging.pipeline_tag_from_pipeline_settings()

    # Unlike other features, the redshifts of the lens and source do not change the settings tag and phase path. Thus,
    # our output will simply go to the phase path:

    # phase_path = 'phase_name/settings'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag//'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return al.Mask.circular_annular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.2,
            outer_radius_arcsec=3.3,
        )

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use the input value of redshifts from the pipeline.

    class LensSourceX1Phase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
            self.galaxies.lens.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(
        phase_name="phase_1__x1_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source_0=al.GalaxyModel(
                redshift=redshift_source, light=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        optimizer_class=af.MultiNest,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Use manually specified new values of redshifts for the lens and source galaxies.

    class LensSourceX2Phase(al.PhaseImaging):
        def customize_priors(self, results):

            self.galaxies.lens = results.from_phase(
                "phase_1__x1_source"
            ).variable.galaxies.lens

            self.galaxies.source = results.from_phase(
                "phase_1__x1_source"
            ).variable.galaxies.source

    phase2 = LensSourceX2Phase(
        phase_name="phase_2__x2_source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=1.0,
                mass=al.mass_profiles.EllipticalIsothermal,
                shear=al.mass_profiles.ExternalShear,
            ),
            source=al.GalaxyModel(
                redshift=2.0, light=al.light_profiles.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return al.PipelineImaging(pipeline_name, phase1, phase2)
