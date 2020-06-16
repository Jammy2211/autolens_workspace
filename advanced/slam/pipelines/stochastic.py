import autofit as af
import autolens as al

"""
In this pipeline, we fit the mass of a strong lens using a power-law + shear model.

The mass model and source are initialized using an already run 'source' pipeline.

The pipeline is one phases:

Phase 1:

Fit the lens mass model as a power-law, using the source model from a previous pipeline.
Lens Mass: EllipticalPowerLaw + ExternalShear
Source Light: Previous Pipeline Source.
Previous Pipeline: no_lens_light/source/*/lens_sie__source_*py
Prior Passing: Lens Mass (model -> previous pipeline), source (model / instance -> previous pipeline)
Notes: If the source is parametric, its parameters are varied, if its an inversion, they are fixed.
"""


def source_is_inversion_from_slam(slam):
    if slam.source.type_tag in "sersic":
        return False
    else:
        return True


def source_with_previous_model_or_instance(slam):
    """slam the source source model using the previous pipeline or phase results.

    This function is required because the source light model is not specified by the pipeline itself (e.g. the previous
    pipelines determines if the source was modeled using parametric *LightProfile*s or an inversion.

    If the source was parametric this function returns the source as a model, given that a parametric source should be
    fitted for simultaneously with the mass model.

    If the source was an inversion then it is returned as an instance, given that the inversion parameters do not need
    to be fitted for alongside the mass model.

    The bool include_hyper_source determines if the hyper-galaxy used to scale the sources noises is included in the
    model fitting.
    """

    if slam.hyper.hyper_galaxies:

        hyper_galaxy = af.PriorModel(al.HyperGalaxy)

        hyper_galaxy.noise_factor = (
            af.last.hyper_combined.model.galaxies.source.hyper_galaxy.noise_factor
        )
        hyper_galaxy.contribution_factor = (
            af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = (
            af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.noise_power
        )

    else:

        hyper_galaxy = None

    if slam.source.type_tag in "sersic":

        return al.GalaxyModel(
            redshift=af.last.instance.galaxies.source.redshift,
            sersic=af.last.model.galaxies.source.sersic,
            hyper_galaxy=hyper_galaxy,
        )

    else:

        return al.GalaxyModel(
            redshift=af.last.instance.galaxies.source.redshift,
            pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
            regularization=af.last.hyper_combined.instance.galaxies.source.regularization,
            hyper_galaxy=hyper_galaxy,
        )


def make_pipeline(
    slam,
    phase,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    settings=al.PhaseSettingsImaging(),
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_mass__power_law__stochastic"

    """TAG: Setup the lens mass tag for pipeline tagging"""
    slam.set_mass_type(mass_type="power_law")

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting settings (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(slam.hyper.tag)
    phase_folders.append(slam.source.tag)
    phase_folders.append(slam.mass.tag)

    """SLaM: Set whether shear is Included in the mass model."""

    if not slam.mass.no_shear:
        if af.last.model.galaxies.lens.shear is not None:
            shear = af.last.model.galaxies.lens.shear
        else:
            shear = al.mp.ExternalShear
    else:
        shear = None

    ### PHASE 1 ###

    # In phase 1, we fit the lens galaxy's mass and source, where we:

    # 1) Use the source galaxy of the 'source' pipeline.
    # 2) Set priors on the lens galaxy mass using the EllipticalIsothermal and ExternalShear of previous pipelines.

    # slam the power-law *MassProfile* and initialize its priors from the SIE.

    mass = af.PriorModel(al.mp.EllipticalPowerLaw)

    mass.centre = af.last.model.galaxies.lens.mass.centre
    mass.elliptical_comps = af.last.model.galaxies.lens.mass.elliptical_comps
    mass.einstein_radius = af.last.model_absolute(
        a=0.3
    ).galaxies.lens.mass.einstein_radius

    # slam the source model, which uses a variable parametric profile or fixed inversion model depending on the
    # previous pipeline.

    source = source_with_previous_model_or_instance(slam=slam)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_power_law__source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(redshift=redshift_lens, mass=mass, shear=shear),
            source=source,
        ),
        settings=settings,
        non_linear_class=af.DynestyStatic(),
    )

    return al.PipelineDataset(pipeline_name, phase1)