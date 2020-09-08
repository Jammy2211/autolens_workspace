import autofit as af
import autolens as al

"""
This pipeline performs an analysis which fits an image with the lens light included, and a source galaxy
using an _Inversion_, using a decomposed light and dark matter profile. The pipeline follows on from the
inversion pipeline 'pipelines/with_lens_light/inversion/from_source__parametric/lens_light_sie__source_inversion.py'.

Alignment of the centre, phi and axis-ratio of the _LightProfile_'s _EllipticalSersic_ and EllipticalExponential
profiles use the alignment specified in the previous pipeline.

The pipeline is two phases:

Phase 1:

    Description: Fit the lens light and mass model as a decomposed profile, using an _Inversion_ for the Source.
    Lens Light & Mass: _EllipticalSersic_ + EllipticalExponential
    Lens Mass: SphericalNFW + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: with_lens_light/bulge_disk/inversion/from_source__parametric/light_bulge_disk_sieexp_source_inversion.py
    Prior Passing: Lens Light (model -> previous pipeline), Lens Mass (default),
                   Source _Inversion_ (model / instance -> previous pipeline)
    Notes: Uses an interpolation pixel scale for fast _EllipticalPowerLaw_ deflection angle calculations by default.

Phase 2:
    
    Description: Refines the _Inversion_ parameters, using a fixed mass model from phase 1.
    Lens Light & Mass: EllipticalSersic
    Lens Mass: SphericalNFW + ExternalShear
    Source Light: VoronoiMagnification + Constant
    Previous Pipelines: None
    Prior Passing: Lens Light & Mass (instance -> phase 1), source _Inversion_ (model -> phase 1)
    Notes: Uses an interpolation pixel scale for fast _EllipticalPowerLaw_ deflection angle calculations by default.
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__light_dark__bulge_disk"

    """TAG: Setup the lens mass tag for pipeline tagging"""
    slam.set_mass_type(mass_type="light_dark")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The bulge + disk centres or elliptical_comps are aligned.
        3) The disk component of the lens light model is an _EllipticalExponential_ or _EllipticalSersic_ profile.
        4) The lens galaxy mass model includes an  _ExternalShear_.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.setup_hyper.tag,
        slam.setup_source.tag,
        slam.pipeline_light.tag,
        slam.pipeline_mass.tag,
    ]

    """SLaM: Set whether shear is Included in the mass model."""

    shear = slam.pipeline_mass.shear_from_previous_pipeline

    """
    Phase 1: Fit the lens galaxy's light and mass and one source galaxy, where we:

        1) Fix the lens galaxy's light using the the _LightProfile_ inferred in the previous 'light' pipeline, including
           assumptions related to the geometric alignment of different components.
        2) Pass priors on the lens galaxy's SphericalNFW _MassProfile_'s centre using the EllipticalIsothermal fit of the
           previous pipeline, if the NFW centre is a free parameter.
        3) Pass priors on the lens galaxy's shear using the ExternalShear fit of the previous pipeline.
        4) Pass priors on the source galaxy's light using the _EllipticalSersic_ of the previous pipeline.
    """

    """SLaM: Set if the Sersic bulge is modeled with or without a mass-to-light gradient."""

    bulge = slam.pipeline_mass.bulge_light_and_mass_profile

    """SLaM: Set if the disk is modeled as an _EllipticalSersic_ or _EllipticalExponential_ with or without a mass-to-light gradient."""

    disk = slam.pipeline_mass.disk_light_and_mass_profile

    """SLaM: Set all the mass-to-light ratios of all light and mass profiles to the same value, if set as constant."""

    slam.pipeline_mass.set_mass_to_light_ratios_of_light_and_mass_profiles(
        light_and_mass_profiles=[bulge, disk]
    )

    """SLaM: Include a Super-Massive Black Hole (SMBH) in the mass model is specified in _SLaMPipelineMass_."""

    smbh = slam.pipeline_mass.smbh_from_centre(
        centre=af.last.instance.galaxies.lens.bulge.centre
    )

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
        dark=al.mp.SphericalNFWMCRLudlow,
        shear=shear,
        smbh=smbh,
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

    if slam.pipeline_light.disk_as_sersic:
        lens.disk.sersic_index = af.last.instance.galaxies.lens.disk.sersic_index

    if slam.pipeline_mass.align_bulge_dark_centre:
        lens.dark.centre = lens.bulge.centre
    else:
        lens.dark.centre = af.last.model.galaxies.lens.bulge.centre

    lens.dark.mass_at_200 = af.LogUniformPrior(lower_limit=5e8, upper_limit=5e14)

    lens.dark.redshift_object = slam.redshift_lens
    lens.dark.slam.redshift_source = slam.redshift_source

    source = slam.source_from_source_pipeline_for_mass_pipeline()

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_light_mlr_nfw__source__fixed_lens_light",
        folders=folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=30),
    )

    """
    Phase 2: Fit the lens galaxy's light and mass and source galaxy using the results of phase 1 as
    initialization
    """

    bulge = slam.pipeline_mass.bulge_light_and_mass_profile
    disk = slam.pipeline_mass.disk_light_and_mass_profile

    lens = al.GalaxyModel(
        redshift=slam.redshift_lens,
        bulge=bulge,
        disk=disk,
        dark=phase1.result.model.galaxies.lens.dark,
        shear=af.last[-2].model.galaxies.lens.shear,
        smbh=phase1.result.model.galaxies.lens.smbh,
        hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
    )

    lens.bulge.centre = af.last[-1].model.galaxies.lens.bulge.centre
    lens.bulge.elliptical_comps = af.last[-1].model.galaxies.lens.bulge.elliptical_comps
    lens.bulge.intensity = af.last[-1].model.galaxies.lens.bulge.intensity
    lens.bulge.effective_radius = af.last[-1].model.galaxies.lens.bulge.effective_radius
    lens.bulge.sersic_index = af.last[-1].model.galaxies.lens.bulge.sersic_index

    lens.bulge.mass_to_light_ratio = (
        phase1.result.model.galaxies.lens.bulge.mass_to_light_ratio
    )

    lens.disk.centre = af.last[-1].model.galaxies.lens.disk.centre
    lens.disk.elliptical_comps = af.last[-1].model.galaxies.lens.disk.elliptical_comps
    lens.disk.intensity = af.last[-1].model.galaxies.lens.disk.intensity
    lens.disk.effective_radius = af.last[-1].model.galaxies.lens.disk.effective_radius
    if slam.pipeline_light.disk_as_sersic:
        lens.disk.sersic_index = af.last[-1].model.galaxies.lens.disk.sersic_index

    lens.disk.mass_to_light_ratio = (
        phase1.result.model.galaxies.lens.disk.mass_to_light_ratio
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2__lens_light_mlr_nfw__source",
        folders=folders,
        galaxies=dict(lens=lens, source=phase1.result.model.galaxies.source),
        hyper_image_sky=phase1.result.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=phase1.result.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    if not slam.setup_hyper.hyper_fixed_after_source:

        phase2 = phase2.extend_with_multiple_hyper_phases(
            setup_hyper=slam.setup_hyper, include_inversion=True
        )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
