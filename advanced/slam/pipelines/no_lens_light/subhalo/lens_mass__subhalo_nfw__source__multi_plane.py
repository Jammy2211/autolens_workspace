import autofit as af
import autolens as al

"""
In this pipeline, we'll perform a subhalo analysis which determines the attempts to detect subhalos by putting
subhalos at fixed intevals on a 2D (y,x) grid.

The mass model and source are initialized using an already run 'source' and 'mass' pipeline.

The pipeline is as follows:

Phase 1:

    Perform the subhalo detection analysis.
    
    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Subhalo: SphericalNFWLudlow
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass (instance -> previous pipeline), source light (model -> previous pipeline).
    Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

Phase 2:

    Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
    
    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Subhalo: SphericalNFWLudlow
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass & source light (model -> previous pipeline), subhalo mass (model -> phase 2).
    Notes: None
"""


def make_pipeline(
    slam,
    settings,
    redshift_lens=0.5,
    redshift_source=1.0,
    number_of_steps=5,
    parallel=False,
):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_subhalo__nfw"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an external shear.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.hyper.tag,
        slam.source.tag,
        slam.mass.tag,
    ]

    """
    Phase 1: Attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

        1) The lens model and source parameters are held fixed to the best-fit values of the previous pipeline.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
    """

    class GridPhase(af.as_grid_search(phase_class=al.PhaseImaging, parallel=parallel)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(redshift=redshift_lens, mass=al.mp.SphericalNFWMCRLudlow)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)

    """
    SLaM: Setup the source model, which uses a variable parametric profile or fixed inversion model.
    """

    source = slam.source_from_source_pipeline_for_mass_pipeline()

    subhalo.mass.redshift_source = source.redshift

    """The subhalo redshift is free to vary between 0.0 and the lens galaxy redshift."""

    subhalo.mass.redshift_object = af.UniformPrior(
        lower_limit=0.0, upper_limit=subhalo.redshift
    )

    phase1a = GridPhase(
        phase_name="phase_1a__subhalo_search__z_below_lens__source",
        folders=folders,
        galaxies=dict(
            lens=af.last.instance.galaxies.lens, subhalo=subhalo, source=source
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=50,
            evidence_tolerance=slam.source.inversion_evidence_tolerance,
        ),
        number_of_steps=number_of_steps,
    )

    """The subhalo redshift is free to vary between 0.0 and the lens galaxy redshift."""

    subhalo.mass.redshift_object = af.UniformPrior(
        lower_limit=subhalo.redshift, upper_limit=source.redshift
    )

    phase1b = GridPhase(
        phase_name="phase_1b__subhalo_search__z_above_lens__source",
        folders=folders,
        galaxies=dict(
            lens=af.last.instance.galaxies.lens, subhalo=subhalo, source=source
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(
            n_live_points=50,
            evidence_tolerance=slam.source.inversion_evidence_tolerance,
        ),
        number_of_steps=5,
    )

    return al.PipelineDataset(pipeline_name, phase1a, phase1b)
