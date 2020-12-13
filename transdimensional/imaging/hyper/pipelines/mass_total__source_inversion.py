from os import path
import autofit as af
import autolens as al

"""
HYPER PIPELINE INTERFACE 

This pipeline uses PyAutoLens`s hyper-features and descriptions of these features is given below. Hyper-mode itself
is described fully in chapter 5 of the HowToLens lecture series and I recommend you follow those tutorials first to
gain a clear understanding of hyper-mode.

HYPER MODEL OBJECTS 

Below you`ll note the following three hyper-model objects:
    
 - hyper_galaxy - If used, the noise-map in the bright regions of the galaxy is scaled.
 - hyper_image_sky - If used, the background sky of the image being fitted is included as part of the model.
 - hyper_background_noise - If used, the background noise of the noise-map is included as part of the model.

An example of these objects being used to make a phase is as follows:

phase = al.PhaseImaging(
    name="phase___hyper_example",
    
    galaxies=af.CollectionPriorModel(
        lens=al.GalaxyModel(
            redshift=setup.redshift_lens,
            hyper_galaxy=phase_last.result.hyper.instance.optional.galaxies.lens.hyper_galaxy,
        ),
        source=al.GalaxyModel(
            redshift=setup.redshift_source,
        ),
    ),
    hyper_image_sky=phase_last.result.hyper.instance.optional.hyper_image_sky,
    hyper_background_noise=phase_last.result.hyper.instance.optional.hyper_background_noise,
    search=af.DynestyStatic(),
)

Above, we pass inferred hyper model components to the phase (the `hyper_combined` attribute is described next).

What does the `optional` attribute mean? It means that the component is only passed if it is used. For example, if
hyper_image_sky is turned off (by settting hyper_image_sky to `False` in the PipelineGeneralSettings), this model
component will not be passed. That is, it is optional.

__HYPER PHASES__

The hyper-galaxies, hyper-image_sky and hyper-background-noise all have non-linear parameters we need to fit for
during our modeling.

How do we fit for the hyper-parameters using our `NonLinearSearch` (e.g. MultiNest)? Typically, we don't fit for them
simultaneously with the lens and source models, as this creates an unnecessarily large parameter space which we`d
fail to fit accurately and efficiently.

Instead, we `extend` phases with extra phases that specifically fit certain components of hyper-galaxy-model. You`ve
hopefully already seen the following code, which optimizes just the parameters of an `Inversion` (e.g. the
pixelization and regularization):

    phase1 = phase1.extend_with_inversion_phase()

Extending a phase with hyper phases is just as easy:

    phase = phase.extend_with_multiple_hyper_phases(
            search=af.DynestyStatic(),
            inversion=True,
        hyper-galaxy=True,
        include_background_sky=True,
        include_background_noise=True,
    )

This extends the phase with 3 additional phases which:

    1) Fit the `Inversion` parameters using the `Pixelization` and `Regularization` scheme that were used in the main phase.
       (e.g. a brightness-based `Pixelization` and adaptive `Regularization` scheme). The best-fit lens and source
       models are used. This is called the `inversion` phase.
    
    2) Simultaneously fit the hyper-galaxies, background sky and background noise hyper parameters using the best-fit
       lens and source models from the main phase. This phase only scales the noise and the image. This is called
       the `hyper-galaxy` phase.
    
    3) Fit all of the components above using Gaussian priors centred on the resulting values of phases 1) and 2). This 
       is important as there is a trade-off between increasing the noise in the lens / source and changing the 
       `Pixelization` `Regularization`.hyper-galaxy-parameters. This is called the `hyper_combined` phase.

Above, we used the results of the `hyper_combined` phase to setup the hyper-galaxies, hyper_image_sky, and
hyper_background_noise. Typically, we set these components up as `instances` whose parameters are fixed during the
main phases which fit the lens and source models.

PIPELINE DESCRIPTION 

In this pipeline, we fit the a strong lens using a `EllipticalIsothermal` `MassProfile`.and a source which uses an 
inversion. The pipeline will use hyper-features, that adapt the `Inversion` and other aspects of the model to the data
being fitted.

The pipeline is as follows:

Phase 1:

    Fit the lens mass model and source `LightProfile`.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the inversion`s `Pixelization` and `Regularization`, using a magnification
    based pixel-grid and the previous lens mass model.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Mass (instance -> phase 1).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Phase 3:
    
    Refine the lens mass model using the source `Inversion`.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Prior Passing: Lens Mass (model -> previous pipeline), source `Inversion` (instance -> phase 2).
    Notes: Lens mass varies, source `Inversion` parameters fixed.

Phase 4:

    Fit the inversion`s `Pixelization` and `Regularization`, using the input pixelization,
    `Regularization` and the previous lens mass model.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: setup.source.pixelization + setup.source.regularization
    Prior Passing: Lens Mass (instance -> phase 3).
    Notes:  Lens mass fixed, source `Inversion` parameters vary.

Phase 5:
    
    Refine the lens mass model using the `Inversion`.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: setup.source.pixelization + setup.source.regularization
    Prior Passing: Lens Mass (model -> phase 3), source `Inversion` (instance -> phase 4).
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass[total]_source[inversion]"

    """
    This pipeline is tagged according to whether:

    1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    2) The lens galaxy mass model includes an `ExternalShear`.
    3) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in phases 4 & 5).
    """

    path_prefix = path.join(setup.path_prefix, pipeline_name, setup.tag)

    """
    Phase 1: Fit the lens`s `MassProfile`'s and source `LightProfile`, where we:

        1) Use an `EllipticalIsothermal` for the lens's mass and `EllipticalSersic`for the source's bulge, 
           irrespective of the final model that is fitted by the pipeline.
        2) Include an `ExternalShear` in the mass model if `SetupMass.with_shear=True`.
        3) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    """

    phase1 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[1]_mass[sie]_source[bulge]", n_live_points=80
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=al.mp.EllipticalIsothermal,
                shear=setup.setup_mass.shear_prior_model,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, bulge=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
    )

    """Extends the phase with hyper-phases using `SetupHyper` inputs, as described at the top of this example script."""

    phase1 = phase1.extend_with_hyper_phase(setup_hyper=setup.setup_hyper)

    """
    Phase 2:

    Phases 2 & 3 use a magnification based `Pixelization` and constant `Regularization` scheme to reconstruct the source.
    The `SetupSourceInversion.pixelization_prior_model` & `SetupSourceInversion.regularization_prior_model` are not 
    used until phases 4 & 5.

    This is because a `Pixelization` / `Regularization` that adapts to the source`s surface brightness uses a previous
    model image of that source (its `hyper-image`). If the source`s true morphology is irregular, or there are
    multiple sources, the `EllipticalSersic` bulge used in phase 1 would give a poor hyper-image. In contrast, the
    `Inversion` below will accurately capture such a source.

    In phase 2, we fit the `Pixelization` and `Regularization`, where we:

        1) Fix the lens mass model to the mass-model inferred by the previous pipeline.
        2) Use a `VoronoiMagnification` `Pixelization`.
        3) Use a `Constant` `Regularization`.
    """

    phase2 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[2]_mass[fixed]_source[inversion_magnification_initialization]",
            n_live_points=20,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.instance.galaxies.lens.mass,
                shear=phase1.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=phase1.result.hyper.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    phase2 = phase2.extend_with_hyper_phase(setup_hyper=setup.setup_hyper)

    """
    Phase 3: Refit the lens`s mass and source galaxy using the magnification `Inversion`, where we:

        1) Fix the source `Inversion` parameters to the results of phase 2.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of phase 2.
    """

    phase3 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[3]_mass[sie]_source[inversion_magnification]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase2.result.instance.galaxies.source.pixelization,
                regularization=phase2.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase2.result.hyper.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    phase3 = phase3.extend_with_hyper_phase(setup_hyper=setup.setup_hyper)

    """
    Phase 4: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens`s `MassProfile`'s to the results of phase 3.
        2) Use the `Pixelization` input into `SetupSourceInversion.pixelization_prior_model`.
        3) Use the `Regularization` input into `SetupSourceInversion.regularization_prior_model`.
    """

    phase4 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[4]_mass[fixed]_source[inversion_initialization]",
            n_live_points=20,
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase3.result.instance.galaxies.lens.mass,
                shear=phase3.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=setup.setup_source.pixelization_prior_model,
                regularization=setup.setup_source.regularization_prior_model,
                hyper_galaxy=phase3.result.hyper.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    phase4 = phase4.extend_with_hyper_phase(
        setup_hyper=setup.setup_hyper,
        inversion_pixels_fixed=setup.setup_source.inversion_pixels_fixed,
    )

    """
    Phase 5: Fit the lens`s mass using the input pipeline `SetupSourceInversion.pixelization_prior_model` & 
    `SetupSourceInversion.regularization_prior_model` where we:

        1) Fix the source `Inversion` parameters to the results of phase 4.
        2) Set priors on the lens galaxy `MassProfile`'s using the results of phase 3.
    """

    phase5 = al.PhaseImaging(
        search=af.DynestyStatic(
            name="phase[5]_mass[sie]_source[inversion]", n_live_points=50
        ),
        galaxies=af.CollectionPriorModel(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                mass=phase3.result.model.galaxies.lens.mass,
                shear=phase3.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                pixelization=phase4.result.instance.galaxies.source.pixelization,
                regularization=phase4.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase4.result.hyper.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase4.result.hyper.instance.optional.hyper_image_sky,
        hyper_background_noise=phase4.result.hyper.instance.optional.hyper_background_noise,
        settings=settings,
    )

    return al.PipelineDataset(
        pipeline_name, path_prefix, phase1, phase2, phase3, phase4, phase5
    )
