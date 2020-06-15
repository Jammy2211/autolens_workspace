import autofit as af
import autolens as al

"""
In this pipeline, we fit the lens light of a strong lens using a two component bulge + disk model.

The mass model and source are initialized using an already run 'source' pipeline. Although the lens light was
fitted in this pipeline, we do not use this model to set priors in this pipeline.

The bulge and disk are modeled using EllipticalSersic and EllipticalExponential profiles respectively. Their alignment
(centre, elliptical components) and whether the disk component is instead modeled using an EllipticalSersic profile
can be customized using the pipeline slam.

The pipeline is one phase:

Phase 1:

Fit the lens light using a bulge + disk model, with the lens mass and source fixed to the
result of the previous pipeline

Lens Light & Mass: EllipticalSersic + EllipticalExponential
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: Previous 'source' pipeline.
Previous Pipelines: with_lens_light/source/*/lens_bulge_disk_sie__source_*.py
Prior Passing: Lens Mass (instance -> previous pipeline), Source (instance -> previous pipeliine).
Notes: Can be customized to vary the lens mass and source.
"""


def make_pipeline(
    slam, phase_folders=None, redshift_lens=0.5, settings=al.PhaseSettingsImaging()
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_light__bulge_disk_x2"

    """TAG: Setup the lens light tag for pipeline tagging"""
    slam.set_light_type(light_type="bulge_disk_x2")

    """
    This pipeline is tagged according to whether:

    1) Hyper-fitting settings (galaxies, sky, background noise) are used.
    2) The bulge + disk centres, rotational angles or axis ratios are aligned.
    3) The disk component of the lens light model is an Exponential or Sersic profile.
    4) The lens galaxy mass model includes an external shear.
    """

    phase_folders.append(pipeline_name)
    phase_folders.append(slam.hyper.tag)
    phase_folders.append(slam.source.tag)
    phase_folders.append(slam.light.tag)

    """
    Phase 1: Fit the lens galaxy's light, where we:

    1) Fix the lens galaxy's mass and source galaxy to the results of the previous pipeline.
    2) Vary the lens galaxy hyper noise factor if hyper-galaxies noise scaling is on.

    If hyper-galaxy noise scaling is on, it may over-scale the noise making this new *LightProfile* fit the data less
    well. This is circumvented by including the noise scaling as a free parameter.
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

    """SLaM: Set whether the disk is modeled as a Sersic or Exponential."""

    if slam.light.disk_as_sersic:
        disk = af.PriorModel(al.lp.EllipticalSersic)
        disk_1 = af.PriorModel(al.lp.EllipticalSersic)
    else:
        disk = af.PriorModel(al.lp.EllipticalExponential)
        disk_1 = af.PriorModel(al.lp.EllipticalExponential)

    bulge = af.PriorModel(al.lp.EllipticalSersic)

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=bulge,
        disk=disk,
        disk_1=disk_1,
        mass=af.last.instance.galaxies.lens.mass,
        shear=af.last.instance.galaxies.lens.shear,
        hyper_galaxy=hyper_galaxy,
    )

    """SLaM: Set the alignment of the bulge and disk's centres, phis and axis-ratios."""

    if slam.light.align_bulge_disk_centre:
        lens.bulge.centre = lens.disk.centre

    if slam.light.align_bulge_disk_elliptical_comps:
        lens.bulge.elliptical_comps = lens.disk.elliptical_comps

    source = slam.source_from_previous_pipeline()

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_bulge_disk_sie__source",
        phase_folders=phase_folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=50, sampling_efficiency=0.5, evidence_tolerance=0.8
        ),
    )

    if not slam.hyper.hyper_fixed_after_source:

        phase1 = phase1.extend_with_multiple_hyper_phases(
            hyper_galaxy_search=slam.hyper.hyper_galaxies_search,
            inversion_search=slam.hyper.inversion_search,
            hyper_combined_search=slam.hyper.hyper_combined_search,
            include_background_sky=slam.hyper.hyper_image_sky,
            include_background_noise=slam.hyper.hyper_background_noise,
        )

    return al.PipelineDataset(pipeline_name, phase1)
