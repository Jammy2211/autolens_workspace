import autofit as af
import autolens as al

"""
This pipeline performs a parametric source analysis which fits an image with a lens mass model and
source galaxy.

The pipeline is as follows:

Phase 1:

    Fit the lens mass model and source _LightProfile_.
    
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
        2) The lens galaxy mass model includes an _ExternalShear_.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.setup_hyper.tag,
        slam.source_parametric_tag,
    ]

    """
    Phase 1: Fit the lens's _MassProfile_'s and source galaxy.
    """

    mass = slam.pipeline_source_parametric.setup_mass.mass_profile

    """SLaM: Align the mass model centre with the input slam value, if input."""

    mass = slam.pipeline_source_parametric.setup_mass.align_centre_to_mass_centre(
        mass=mass
    )

    """SLaM: The shear model is chosen below based on the settings of the slam source."""

    phase1 = al.PhaseImaging(
        phase_name="phase_1__mass_sie__source_sersic",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens,
                mass=mass,
                shear=slam.pipeline_source_parametric.shear,
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
