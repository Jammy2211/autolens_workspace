import autofit as af
import autolens as al

"""
This pipeline performs a parametric source analysis which fits an image with a lens mass model and
source galaxy.

This pipeline uses 1 phase:

Phase 1:

    Fit the lens mass model and source `LightProfile`.
    
    Lens Mass: MassProfile (default=EllipticalIsothermal) + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: None
    Notes: None
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__sersic"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an `ExternalShear`.
    """

    path_prefix = f"{slam.path_prefix}/{pipeline_name}/{slam.source_parametric_tag}"

    """
    Phase 1: Fit the lens`s `MassProfile`'s and source galaxy.
    """

    mass = af.PriorModel(slam.pipeline_source_parametric.setup_mass.mass_profile)

    """SLaM: Align the mass model centre with the input slam value, if input."""

    mass = slam.pipeline_source_parametric.setup_mass.align_centre_to_mass_centre(
        mass_prior_model=mass
    )

    """SLaM: The shear model is chosen below based on the settings of the slam source."""

    phase1 = al.PhaseImaging(
        path_prefix=path_prefix,
        phase_name="phase_1__mass_sie__source_sersic",
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=mass,
                shear=slam.pipeline_source_parametric.setup_mass.shear_prior_model,
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=200, walks=10),
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(setup_hyper=slam.setup_hyper)

    return al.PipelineDataset(pipeline_name, phase1)
