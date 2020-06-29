import autofit as af
import autolens as al

"""
In this pipeline, we'll perform an analysis which fits an image with the lens light included, and a source galaxy
using an inversion, using a decomposed light and dark matter profile. The pipeline follows on from the
inversion pipeline 'pipelines/with_lens_light/inversion/from_source__parametric/lens_light_sie__source_inversion.py'.

Alignment of the centre, phi and axis-ratio of the _LightProfile_'s EllipticalSersic and EllipticalExponential
profiles use the alignment specified in the previous pipeline.

The pipeline is two phases:

Phase 1:

    Description: Fit the lens light and mass model as a decomposed profile, using an inversion for the Source.
    Lens Light & Mass: EllipticalSersic + EllipticalExponential
    Lens Mass: SphericalNFW + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: with_lens_light/bulge_disk/inversion/from_source__parametric/lens_bulge_disk_sieexp_source_inversion.py
    Prior Passing: Lens Light (model -> previous pipeline), Lens Mass (default),
                   Source Inversion (model / instance -> previous pipeline)
    Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.

Phase 2:
    
    Description: Refines the inversion parameters, using a fixed mass model from phase 1.
    Lens Light & Mass: EllipticalSersic
    Lens Mass: SphericalNFW + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: None
    Prior Passing: Lens Light & Mass (instance -> phase 1), source inversion (model -> phase 1)
    Notes: Uses an interpolation pixel scale for fast power-law deflection angle calculations by default.
"""


def make_pipeline(slam, settings, redshift_lens=0.5, redshift_source=1.0):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__light_dark__bulge_disk"

    """TAG: Setup the lens mass tag for pipeline tagging"""
    slam.set_mass_type(mass_type="light_dark")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The bulge + disk centres, rotational angles or axis ratios are aligned.
        3) The disk component of the lens light model is an Exponential or Sersic profile.
        4) The lens galaxy mass model includes an external shear.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.hyper.tag,
        slam.source.tag,
        slam.light.tag,
        slam.mass.tag,
    ]

    """SLaM: Set whether shear is Included in the mass model."""

    shear = slam.mass.shear_from_previous_pipeline

    """
    Phase 1: Fit the lens galaxy's light and mass and one source galaxy, where we:

        1) Fix the lens galaxy's light using the the _LightProfile_ inferred in the previous 'light' pipeline, including
           assumptions related to the geometric alignment of different components.
        2) Pass priors on the lens galaxy's SphericalNFW _MassProfile_'s centre using the EllipticalIsothermal fit of the
           previous pipeline, if the NFW centre is a free parameter.
        3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
        4) Pass priors on the source galaxy's light using the EllipticalSersic of the previous pipeline.
    """

    """SLaM: Set whether the disk is modeled as a Sersic or Exponential."""

    if slam.light.disk_as_sersic:
        disk = af.PriorModel(al.lmp.EllipticalSersic)
    else:
        disk = af.PriorModel(al.lmp.EllipticalExponential)

    lens = al.GalaxyModel(
        redshift=redshift_lens,
        bulge=al.lmp.EllipticalSersic,
        disk=disk,
        dark=al.mp.SphericalNFWMCRLudlow,
        shear=shear,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    lens.bulge.centre = af.last.instance.galaxies.lens.bulge.centre
    lens.bulge.elliptical_comps = af.last.instance.galaxies.lens.bulge.elliptical_comps
    lens.bulge.intensity = af.last.instance.galaxies.lens.bulge.intensity
    lens.bulge.effective_radius = af.last.instance.galaxies.lens.bulge.effective_radius
    lens.bulge.sersic_index = af.last.instance.galaxies.lens.bulge.sersic_index

    lens.disk.centre = af.last.instance.galaxies.lens.disk.centre
    lens.disk.elliptical_comps = af.last.instance.galaxies.lens.disk.elliptical_comps
    lens.disk.phi = af.last.instance.galaxies.lens.disk.phi
    lens.disk.intensity = af.last.instance.galaxies.lens.disk.intensity
    lens.disk.effective_radius = af.last.instance.galaxies.lens.disk.effective_radius

    if slam.light.disk_as_sersic:
        lens.disk.sersic_index = af.last.instance.galaxies.lens.disk.sersic_index

    if slam.mass.align_bulge_dark_centre:
        lens.dark.centre = lens.bulge.centre
    else:
        lens.dark.centre = af.last.model.galaxies.lens.bulge.centre

    lens.dark.mass_at_200 = af.LogUniformPrior(lower_limit=1e9, upper_limit=1e14)

    lens.dark.redshift_object = redshift_lens
    lens.dark.redshift_source = redshift_source

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_light_mlr_nfw__source__fixed_lens_light",
        folders=folders,
        galaxies=dict(
            lens=lens,
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
                regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
                hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=30,
            evidence_tolerance=slam.source.inversion_evidence_tolerance,
        ),
    )

    """
    Phase 2: Fit the lens galaxy's light and mass and source galaxy using the results of phase 1 as
    initialization
    """

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_light_mlr_nfw__source_inversion",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                bulge=phase1.result.model.galaxies.lens.bulge,
                disk=phase1.result.model.galaxies.lens.disk,
                dark=phase1.result.model.galaxies.lens.dark,
                shear=phase1.result.model.galaxies.lens.shear,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source,
                pixelization=phase1.result.instance.galaxies.source.pixelization,
                regularization=phase1.result.instance.galaxies.source.regularization,
                hyper_galaxy=phase1.result.hyper_combined.instance.optional.galaxies.source.hyper_galaxy,
            ),
        ),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=100,
            evidence_tolerance=slam.source.inversion_evidence_tolerance,
        ),
    )

    if not slam.hyper.hyper_fixed_after_source:

        phase2 = phase2.extend_with_multiple_hyper_phases(
            setup=slam.hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
