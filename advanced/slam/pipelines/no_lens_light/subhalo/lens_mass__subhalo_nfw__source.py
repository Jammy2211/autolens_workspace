import autofit as af
import autolens as al

"""
In this pipeline, we'll perform a subhalo analysis which determines the attempts to detect subhalos by putting
subhalos at fixed intevals on a 2D (y,x) grid.

The mass model and source are initialized using an already run 'source' and 'mass' pipeline.

The pipeline is as follows:

Phase 1:

    Perform the subhalo detection analysis using a *GridSearch* of non-linear searches.

    Lens Mass: Previous mass pipeline model.
    Subhalo: SphericalNFWLudlow
    Source Light: Previous source pipeilne model.
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass (instance -> previous pipeline), source light (model -> previous pipeline).
    Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

Phase 2:

Refine the best-fit detected subhalo from the previous phase.

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
    subhalo_search,
    redshift_lens=0.5,
    redshift_source=1.0,
    source_as_model=True,
    mass_as_model=True,
    grid_size=5,
    parallel=False,
):
    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_subhalo__nfw"

    if not source_as_model:
        pipeline_name += "__src_fixed"

    if not mass_as_model:
        pipeline_name += "__mass_fixed"

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
    Phase 1: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

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

    subhalo.mass.redshift_object = subhalo.redshift

    """
    SLaM: Setup the source model, which uses a variable parametric profile or fixed inversion model.
    """

    if mass_as_model:

        lens = al.GalaxyModel(
            redshift=redshift_lens,
            mass=af.last.model.galaxies.lens.mass,
            shear=af.last.model.galaxies.lens.shear,
            hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
        )

    else:

        lens = al.GalaxyModel(
            redshift=redshift_lens,
            mass=af.last.instance.galaxies.lens.mass,
            shear=af.last.instance.galaxies.lens.shear,
            hyper_galaxy=af.last.hyper_combined.instance.optional.galaxies.lens.hyper_galaxy,
        )

    source = slam.source_from_previous_pipeline_model_or_instance(
        source_as_model=source_as_model, index=0
    )

    subhalo.mass.redshift_source = redshift_source

    phase1 = GridPhase(
        phase_name="phase_1__subhalo_search__source",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=subhalo_search,
        number_of_steps=grid_size,
    )

    subhalo = al.GalaxyModel(redshift=redshift_lens, mass=al.mp.SphericalNFWMCRLudlow)

    subhalo.mass.mass_at_200 = phase1.result.model.galaxies.subhalo.mass.mass_at_200
    subhalo.mass.centre = phase1.result.model.galaxies.subhalo.mass.centre
    subhalo.mass.redshift_object = redshift_lens

    source = slam.source_from_previous_pipeline_model_or_instance(
        source_as_model=True, index=-1
    )

    subhalo.mass.redshift_source = redshift_source

    phase2 = al.PhaseImaging(
        phase_name="phase_2__subhalo_refine",
        folders=folders,
        galaxies=dict(
            lens=af.last[-1].model.galaxies.lens, source=source, subhalo=subhalo
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    phase2 = phase2.extend_with_multiple_hyper_phases(
        setup=slam.hyper, include_inversion=False
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
