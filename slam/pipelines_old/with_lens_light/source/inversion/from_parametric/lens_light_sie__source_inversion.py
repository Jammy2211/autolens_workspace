import autofit as af
import autolens as al

"""
This pipeline performs a source _Inversion_ analysis which fits an image with a source galaxy and a lens light
component.

The lens light model used depends on the previous 'source/parametric' pipeline that is run. For example, if the
bulge-disk pipeline was used the bulge-disk model will be used in this pipeline.

Phases 1 & 2 first use a magnification based _Pixelization_ and constant _Regularization_ scheme to reconstruct the
source (as opposed to immediately using the _Pixelization_ & _Regularization_ input via the pipeline slam).
This ensures that if the input _Pixelization_ or _Regularization_ scheme uses hyper-images, they are initialized using
a pixelized source-plane, which is key for lens's with multiple or irregular sources.

The pipeline is as follows:

Phase 1:

    Set inversion's _Pixelization_ and _Regularization_, using a magnification
    based pixel-grid and the previous lens light and mass model.
    
    Lens Light: Previous Pipeline.
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: source/parametric/light_bulge_disk__mass_sie__source_parametric.py
    Prior Passing: Lens Light / Mass (instance -> previous pipeline).
    Notes: Lens light & mass fixed, source _Inversion_ parameters vary.

Phase 2:

    Refine the lens mass model using the source _Inversion_.
    
    Lens Light: Previous Pipeline.
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: source/parametric/light_bulge_disk__mass_sie__source_parametric.py
    Prior Passing: Lens Light & Mass (model -> previous pipeline), source _Inversion_ (instance -> phase 1).
    Notes: Lens light fixed, mass varies, source _Inversion_ parameters fixed.

Phase 3:

    Fit the inversion's _Pixelization_ and _Regularization_, using the input pixelization,
    _Regularization_ and the previous lens mass model.
    
    Lens Light: Previous Pipeline.
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: slam.source.pixelization + slam.source.regularization
    Previous Pipelines: None
    Prior Passing: Lens Light & Mass (instance -> phase 2).
    Notes:  Lens light & mass fixed, source _Inversion_ parameters vary.

Phase 4:
    
    Refine the lens mass model using the _Inversion_.
    
    Lens Light: Previous Pipeline.
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: _Pixelization_ + regularization
    Prior Passing: Lens Light & Mass (model -> phase 3), source _Inversion_ (instance -> phase 3).
    Notes: Lens light fixed, mass varies, source _Inversion_ parameters fixed.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__inversion"

    """For pipeline tagging we set the source type."""
    slam.set_source_type(source_type=slam.setup_source.tag_no_underscore)

    """
    This pipeline is tagged according to whether:
    
        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The _Pixelization_ and _Regularization_ scheme of the pipeline (fitted in phases 3 & 4).
        3) The lens light model is fixed during the analysis.
        4) The lens galaxy mass model includes an  _ExternalShear_.
        5) The lens light model used in the previous pipeline.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.source_pipeline_tag + slam.lens_light_tag_for_source_pipeline,
        slam.setup_source.tag,
    ]

    """
    Phase 1: fit the _Pixelization_ and _Regularization_, where we:

        1) Fix the lens light & mass model to the light & mass models inferred by the previous pipeline.
    """

    phase1 = al.PhaseImaging(
        phase_name="phase_1__source_inversion_magnification_initialization",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=af.last.instance.galaxies.lens.bulge,
                disk=af.last.instance.galaxies.lens.disk,
                mass=af.last.instance.galaxies.lens.mass,
                shear=af.last.instance.galaxies.lens.shear,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=al.pix.VoronoiMagnification,
                regularization=al.reg.Constant,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=20),
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(
        setup_hyper=slam.setup_hyper, include_inversion=False
    )

    """
    Phase 2: refine the len galaxy mass using an _Inversion_. We will:

        1) Fix the source _Inversion_ parameters to the results of phase 1.
        2) Fix the lens light model to the results of the previous pipeline.
        3) Set priors on the lens galaxy mass from the previous pipeline.
    """

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_light_sie__source_inversion_magnification",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=af.last[-1].instance.galaxies.lens.bulge,
                disk=af.last[-1].instance.galaxies.lens.disk,
                mass=af.last[-1].model.galaxies.lens.mass,
                shear=af.last[-1].model.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    phase2 = phase2.extend_with_multiple_hyper_phases(
        setup_hyper=slam.setup_hyper, include_inversion=False
    )

    """
    Phase 3: Fit the input pipeline _Pixelization_ & _Regularization_, where we:
    
        1) Fix the lens light model to the results of the previous pipeline.
        2) Fix the lens mass model to the mass-model inferred in phase 2.
    """

    phase3 = al.PhaseImaging(
        phase_name="phase_3__source_inversion_initialization",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                bulge=phase2.result.instance.galaxies.lens.bulge,
                disk=phase2.result.instance.galaxies.lens.disk,
                mass=phase2.result.instance.galaxies.lens.mass,
                shear=phase2.result.instance.galaxies.lens.shear,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=slam.setup_source.pixelization,
                regularization=slam.setup_source.regularization,
                hyper_galaxy=phase2.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase2.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase2.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=30,
            evidence_tolerance=slam.setup_hyper.evidence_tolerance,
            sample="rstagger",
        ),
    )

    phase3 = phase3.extend_with_multiple_hyper_phases(
        setup_hyper=slam.setup_hyper, include_inversion=False
    )

    """
    Phase 4: Fit the lens's mass using the input pipeline _Pixelization_ & _Regularization_, where we:

        1) Fix the source _Inversion_ parameters to the results of phase 3.
        2) Fix the lens light model to the results of the previous pipeline.
        3) Set priors on the lens galaxy _MassProfile_'s using the results of phase 2.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    """SLaM: Unfix the lens mass centre if the *mass_centre* was input."""

    mass = slam.setup_source.unfix_mass_centre(mass=mass)

    """SLaM: Unalign the lens mass and light centre, if *align_light_mass_centre* was True."""

    mass = slam.setup_source.unalign_mass_centre_from_light_centre(mass=mass)

    mass.elliptical_comps = phase2.result.model.galaxies.lens.mass.elliptical_comps
    mass.einstein_radius = phase2.result.model.galaxies.lens.mass.einstein_radius

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=phase2.result.instance.galaxies.lens.bulge,
        disk=phase2.result.instance.galaxies.lens.disk,
        mass=mass,
        shear=phase2.result.model.galaxies.lens.shear,
        hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    phase4 = al.PhaseImaging(
        phase_name="phase_4__lens_light_sie__source_inversion",
        folders=folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=slam.redshift_source,
                pixelization=phase3.result.instance.galaxies.source.pixelization,
                regularization=phase3.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase3.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase3.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase3.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    phase4 = phase4.extend_with_multiple_hyper_phases(
        setup_hyper=slam.setup_hyper, include_inversion=True
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3, phase4)
