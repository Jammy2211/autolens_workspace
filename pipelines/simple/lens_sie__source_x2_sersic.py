import autofit as af
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.pipeline.phase import phase_imaging
from autolens.pipeline import pipeline
from autolens.pipeline import pipeline_tagging
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp


# In this pipeline, we'll perform a basic analysis which fits two source galaxies using a parametric light profile and
# a lens galaxy where its light is not present in the image, using two phases:

# Phase 1:

# Description: initialize the lens mass model and source light profile using x1 source.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: None
# Notes: None

# Phase 2:

# Description: Fit the lens mass model and source light profile using x2 sources.
# Lens Mass: EllipitcalIsothermal + ExternalShear
# Source Galaxy 1 - Light: EllipticalSersic
# Source Galaxy 2 - Light: EllipticalSersic
# Previous Pipelines: None
# Prior Passing: Lens mass (variable -> phase 1), Source Galaxy 1 Light (variable -> phase 1)
# Notes: None


def make_pipeline(
    include_shear=True,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    sub_grid_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    positions_threshold=None,
    inner_mask_radii=None,
    interp_pixel_scale=None,
):

    ### SETUP PIPELINE AND PHASE NAMES, TAGS AND PATHS ###

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline_lens_sie__source_x2_sersic"

    pipeline_tag = pipeline_tagging.pipeline_tag_from_pipeline_settings()

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag/'

    phase_folders.append(pipeline_name)
    phase_folders.append(pipeline_tag)

    ### SETUP SHEAR ###

    # If the pipeline should include shear, add this class below so that it enters the phase.

    # After this pipeline this shear class is passed to all subsequent pipelines, such that the shear is either
    # included or omitted throughout the entire pipeline.

    if include_shear:
        shear = mp.ExternalShear
    else:
        shear = None

    # As there is no lens light component, we can use an annular mask throughout this pipeline which removes the
    # central regions of the image.

    def mask_function(image):
        return msk.Mask.circular_annular(
            shape=image.shape,
            pixel_scale=image.pixel_scale,
            inner_radius_arcsec=0.2,
            outer_radius_arcsec=3.3,
        )

    ### PHASE 1 ###

    # In phase 1, we will fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Set our priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.

    class LensSourceX1Phase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            self.galaxies.lens.mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
            self.galaxies.lens.mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = LensSourceX1Phase(
        phase_name="phase_1__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens, mass=mp.EllipticalIsothermal, shear=shear
            ),
            source_0=gm.GalaxyModel(
                redshift=redshift_source, light=lp.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    # You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
    # non-linear parameter space faster (if you haven't already, checkout 'tutorial_7_multinest_black_magic' in
    # 'howtolens/chapter_2_lens_modeling'.

    # Fitting the lens galaxy and source galaxy from uninitialized priors often risks MultiNest getting stuck in a
    # local maxima, especially for the image in this example which actually has two source galaxies. Therefore, whilst
    # I will continue to use constant efficiency mode to ensure fast run time, I've upped the number of live points
    # and decreased the sampling efficiency from the usual values to ensure the non-linear search is robust.

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 80
    phase1.optimizer.sampling_efficiency = 0.2

    ### PHASE 2 ###

    # In phase 2, we will fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Initialize the priors on the lens galaxy using the results of phase 1.
    # 2) Initialize the priors on the first source galaxy using the results of phase 1.

    class LensSourceX2Phase(phase_imaging.PhaseImaging):
        def pass_priors(self, results):

            ## Lens Mass, SIE -> SIE ###

            self.galaxies.lens = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.lens

            ## Source Light, Sersic -> Sersic ###

            self.galaxies.source_0 = results.from_phase(
                "phase_1__lens_sie__source_sersic"
            ).variable.galaxies.source_0

    phase2 = LensSourceX2Phase(
        phase_name="phase_2__lens_sie__source_x2_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=gm.GalaxyModel(
                redshift=redshift_lens, mass=mp.EllipticalIsothermal, shear=shear
            ),
            source_0=gm.GalaxyModel(
                redshift=redshift_source, light=lp.EllipticalSersic
            ),
            source_1=gm.GalaxyModel(
                redshift=redshift_source, light=lp.EllipticalSersic
            ),
        ),
        mask_function=mask_function,
        sub_grid_size=sub_grid_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        positions_threshold=positions_threshold,
        inner_mask_radii=inner_mask_radii,
        interp_pixel_scale=interp_pixel_scale,
        optimizer_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 50
    phase2.optimizer.sampling_efficiency = 0.3

    return pipeline.PipelineImaging(pipeline_name, phase1, phase2)
