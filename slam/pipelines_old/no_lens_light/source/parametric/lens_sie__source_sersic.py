import autofit as af
import autolens as al

"""
This pipeline performs a parametric source analysis which fits an image with a lens mass model and
source galaxy.

The pipeline is as follows:

Phase 1:

    Fit the lens mass model and source _LightProfile_.
    
    Lens Mass: EllipticalIsothermal + ExternalShear
    Source Light: EllipticalSersic
    Previous Pipelines: None
    Prior Passing: None
    Notes: None
"""


def make_pipeline(slam, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__parametric"

    """For pipeline tagging we set the source type."""
    slam.set_source_type(source_type="sersic")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  _ExternalShear_.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.source_pipeline_tag,
        slam.setup_source.tag,
    ]

    """
    Phase 1: Fit the lens's _MassProfile_'s and source galaxy.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    """SLaM: Align the mass model centre with the input slam value, if input."""

    mass = slam.setup_source.align_centre_to_mass_centre(mass=mass)

    """SLaM: The shear model is chosen below based on the settings of the slam source."""

    phase1 = al.PhaseImaging(
        phase_name="phase_1__mass_sie__source_parametric",
        folders=folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=slam.redshift_lens, mass=mass, shear=slam.setup_source.shear
            ),
            source=al.GalaxyModel(
                redshift=slam.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(setup_hyper=slam.setup_hyper)

    return al.PipelineDataset(pipeline_name, phase1)
