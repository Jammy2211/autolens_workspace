import autofit as af
import autolens as al

"""
This pipeline performs a source `Inversion` analysis which fits an image with a lens mass model and
source galaxy.

Phases 1 & 2 use a magnification based `Pixelization` and constant `Regularization` scheme to reconstruct the source
(as opposed to immediately using the `Pixelization` & `Regularization` input via the pipeline slam). This ensures
that if the input `Pixelization` or `Regularization` scheme use hyper-images, they are initialized using
a pixelized source-plane, which is key for lens`s with multiple or irregular sources.

The pipeline is as follows:

Phase 1:

    Fit the inversion`s `Pixelization` and `Regularization`, using a magnification
    based pixel-grid and the previous lens mass model.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: source/parametric/mass_sie__source_parametric.py
    Prior Passing: Lens Mass (instance -> previous pipeline).
    Notes: Lens mass fixed, source `Inversion` parameters vary.

Phase 2:
    
    Refine the lens mass model using the source `Inversion`.

    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: source/parametric/mass_sie__source_parametric.py
    Prior Passing: Lens Mass (model -> previous pipeline), source `Inversion` (instance -> phase 1).
    Notes: Lens mass varies, source `Inversion` parameters fixed.

Phase 3:

    Fit the inversion`s `Pixelization` and `Regularization`, using the input pixelization,
    `Regularization` and the previous lens mass model.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: slam.source.pixelization + slam.source.regularization
    Previous Pipelines: None
    Prior Passing: Lens Mass (instance -> phase 2).
    Notes:  Lens mass fixed, source `Inversion` parameters vary.

Phase 4:

    Refine the lens mass model using the `Inversion`.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: slam.source.pixelization + slam.source.regularization
    Previous Pipelines: source/parametric/mass_sie__source_parametric.py
    Prior Passing: Lens Mass (model -> phase 2), source `Inversion` (instance -> phase 3).
    Notes: Lens mass varies, source `Inversion` parameters fixed.
"""


def make_pipeline(slam, settings, real_space_mask):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__inversion__mass_sie__source_inversion"

    """For pipeline tagging we set the source type."""
    slam.set_source_type(source_type=slam.setup_source.tag_no_underscore)

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
        3) The `Pixelization` and `Regularization` scheme of the pipeline (fitted in phases 3 & 4).
    """

    folders = slam.folders + [
        pipeline_name,
        slam.source_pipeline_tag,
        slam.setup_source.tag,
    ]

    """
    Phase 1 Fit the `Pixelization` and `Regularization`, where we:

        1) Fix the lens mass model to the mass-model inferred by the previous pipeline.
    """

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1__source_inversion_magnification_initialization",
        folders=folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=af.last.instance.galaxies.lens.mass,
                shear=af.last.instance.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
            ),
        ),
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=20),
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(include_inversion=False)

    """
    Phase 2: Fit the lens`s mass and source galaxy using the magnification `Inversion`, where we:
    
        1) Fix the source `Inversion` parameters to the results of phase 1.
        2) Set priors on the lens galaxy `MassProfile``s using the results of the previous pipeline.
    """

    phase2 = al.PhaseInterferometer(
        phase_name="phase_2__mass_sie__source_inversion_magnification",
        folders=folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
            ),
        ),
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    phase2 = phase2.extend_with_multiple_hyper_phases(include_inversion=False)

    """
    Phase 3: Fit the input pipeline `Pixelization` & `Regularization`, where we:

        1) Fix the lens mass model to the mass-model inferred in phase 2.
    """

    phase3 = al.PhaseInterferometer(
        phase_name="phase_3__source_inversion_initialization",
        folders=folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=slam.setup_source.pixelization,
                regularization=slam.setup_source.regularization,
            ),
        ),
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=40),
    )

    phase3 = phase3.extend_with_multiple_hyper_phases(include_inversion=False)

    """
    Phase 4: Fit the lens`s mass using the input pipeline `Pixelization` & `Regularization`, where we:
    
        1) Fix the source `Inversion` parameters to the results of phase 3.
        2) Set priors on the lens galaxy `MassProfile``s using the results of phase 2.
    """

    phase4 = al.PhaseInterferometer(
        phase_name="phase_4__mass_sie__source_inversion",
        folders=folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=phase2.result.model.galaxies.lens.mass,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase3.result.instance.galaxies.source.pixelization,
                regularization=phase3.result.instance.galaxies.source.regularization,
            ),
        ),
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    phase4 = phase4.extend_with_multiple_hyper_phases(include_inversion=True)

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
