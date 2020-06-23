import autofit as af
import autolens as al

"""
In this pipeline, we fit the lens light of a strong lens using multiple elliptical Gaussians.

The mass model and source are initialized using an already run 'source' pipeline. Although the lens light was
fitted in this pipeline, we do not use this model to set priors in this pipeline.

The gaussians are modeled using EllipticalGaussian profiles. Their alignment (centre, elliptical_comps) can be
customized using the pipeline slam.

The pipeline is one phase:

Phase 1:

    Fit the lens light using a multi-Gaussian model, with the lens mass and source fixed to the
    result of the previous pipeline
    
    Lens Light & Mass: EllipticalGaussian(s)
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: Previous 'source' pipeline.
    Previous Pipelines: with_lens_light/source/*/lens_light_sie__source_*.py
    Prior Passing: Lens Mass (instance -> previous pipeline), Source (instance -> previous pipeliine).
    Notes: Can be customized to vary the lens mass and source.
"""


def make_pipeline(
    slam, folders=None, redshift_lens=0.5, settings=al.PhaseSettingsImaging()
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_light__gaussians"

    """TAG: Setup the lens light tag for pipeline tagging"""
    slam.set_light_type(light_type="gaussians")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The number of Gaussians in the lens _LightProfile_.
        3) The lens galaxy mass model includes an external shear.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.hyper.tag,
        slam.source.tag,
        slam.light.tag,
    ]

    """
    Phase 1: Fit the lens galaxy's light, where we:

        1) Fix the lens galaxy's mass and source galaxy to the results of the previous pipeline.
        2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.

    If hyper-galaxy noise scaling is on, it may over-scale the noise making this new _LightProfile_ fit the data less
    well. This can be circumvented by including the noise scaling as a free parameter.
    """

    if slam.hyper.hyper_galaxies:

        hyper_galaxy = af.PriorModel(al.HyperGalaxy)

        hyper_galaxy.noise_factor = (
            af.last.hyper_combined.model.galaxies.lens.hyper_galaxy.noise_factor
        )
        hyper_galaxy.contribution_factor = (
            af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = (
            af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy.noise_power
        )

    else:

        hyper_galaxy = None

    gaussian_0 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_1 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_2 = af.PriorModel(al.lp.EllipticalGaussian)
    gaussian_3 = af.PriorModel(al.lp.EllipticalGaussian)

    gaussian_1.centre = gaussian_0.centre
    gaussian_2.centre = gaussian_0.centre
    gaussian_3.centre = gaussian_0.centre

    if slam.source.lens_light_centre is not None:
        gaussian_0.centre = slam.source.lens_light_centre
        gaussian_1.centre = slam.source.lens_light_centre
        gaussian_2.centre = slam.source.lens_light_centre
        gaussian_3.centre = slam.source.lens_light_centre

    gaussian_0.add_assertion(gaussian_0.sigma < gaussian_1.sigma)
    gaussian_0.add_assertion(gaussian_1.sigma < gaussian_2.sigma)
    gaussian_0.add_assertion(gaussian_2.sigma < gaussian_3.sigma)

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        gaussian_0=gaussian_0,
        gaussian_1=gaussian_1,
        gaussian_2=gaussian_2,
        gaussian_3=gaussian_3,
        mass=af.last.instance.galaxies.lens.mass,
        shear=af.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    source = slam.source_from_previous_pipeline()

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_gaussians_sie__source",
        folders=folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=50, facc=0.5, evidence_tolerance=0.8),
    )

    if not slam.hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            setup=slam.hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1)
