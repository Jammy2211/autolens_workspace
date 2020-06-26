import autofit as af
import autolens as al

"""
In this pipeline, we'll perform a parametric source analysis which fits an image with a lens mass model and
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


def make_pipeline(
    slam,
    settings,
    real_space_mask,
    redshift_lens=0.5,
    redshift_source=1.0,
    evidence_tolerance=5.0,
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_source__parametric"

    """For pipeline tagging we set the source type."""
    slam.set_source_type(source_type="sersic")

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an external shear.
    """

    folders = slam.folders + [pipeline_name, slam.source_pipeline_tag, slam.source.tag]

    """
    Phase 1: Fit the lens galaxy's mass and source galaxy.
    """

    """SLaM: The shear model is chosen below based on the settings of the slam source."""

    phase1 = al.PhaseInterferometer(
        phase_name="phase_1__lens_sie__source_sersic",
        folders=folders,
        real_space_mask=real_space_mask,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens,
                mass=al.mp.EllipticalIsothermal,
                shear=slam.source.shear,
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic,
    )

    phase1.search.n_live_points = 80
    phase1.search.evidence_tolerance = evidence_tolerance

    return al.PipelineDataset(pipeline_name, phase1)
