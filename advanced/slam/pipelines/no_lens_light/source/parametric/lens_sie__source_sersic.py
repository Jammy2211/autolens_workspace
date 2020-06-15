import autofit as af
import autolens as al

"""
In this pipeline, we'll perform a parametric source analysis which fits an image with a lens mass model and
source galaxy.

The pipeline is as follows:

Phase 1:

Fit the lens mass model and source *LightProfile*.

Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: EllipticalSersic
Previous Pipelines: None
Prior Passing: None
Notes: None
"""


def make_pipeline(
    slam,
    settings,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    evidence_tolerance=100.0,
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

    phase_folders.append(pipeline_name)
    phase_folders.append(slam.source_pipeline_tag)
    phase_folders.append(slam.source.tag)

    """
    Phase 1: Fit the lens galaxy's mass and source galaxy.
    """

    mass = af.PriorModel(al.mp.EllipticalIsothermal)

    """SLaM: Align the mass model centre with the input slam value, if input."""

    mass = slam.source.align_centre_to_lens_mass_centre(mass=mass)

    """SLaM: The shear model is chosen below based on the settings of the slam source."""

    phase1 = al.PhaseImaging(
        phase_name="phase_1__lens_sie__source_sersic",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=mass, shear=slam.source.shear
            ),
            source=al.GalaxyModel(
                redshift=redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=80,
            sampling_efficiency=0.2,
            evidence_tolerance=evidence_tolerance,
        ),
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxy_search=slam.hyper.hyper_galaxies_search,
        hyper_combined_search=slam.hyper.hyper_combined_search,
        include_background_sky=slam.hyper.hyper_image_sky,
        include_background_noise=slam.hyper.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1)
