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

    """The subhalo redshift is free to vary between 0.0 and the lens galaxy redshift."""

    subhalo_z_below = al.GalaxyModel(
        redshift=redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo_z_below.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo_z_below.mass.centre_0 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_below.mass.centre_1 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_below.mass.redshift_source = redshift_source
    subhalo_z_below.mass.redshift_object = af.UniformPrior(
        lower_limit=0.0, upper_limit=subhalo_z_below.redshift
    )

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

    source = slam.source_from_source_pipeline_for_mass_pipeline()

    phase1a = GridPhase(
        phase_name="phase_1__subhalo_search__source__z_below_lens",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo_z_below, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=subhalo_search,
        number_of_steps=grid_size,
    )

    """The subhalo redshift is free to vary between and the lens and source galaxy redshifts."""

    subhalo_z_above = al.GalaxyModel(
        redshift=redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo_z_above.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo_z_above.mass.centre_0 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_above.mass.centre_1 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_above.mass.redshift_source = redshift_source
    subhalo_z_above.mass.redshift_object = af.UniformPrior(
        lower_limit=subhalo_z_above.redshift, upper_limit=redshift_source
    )

    phase1b = GridPhase(
        phase_name="phase_1__subhalo_search__source__z_above_lens",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo_z_above, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=subhalo_search,
        number_of_steps=5,
    )

    return al.PipelineDataset(pipeline_name, phase1a, phase1b)
