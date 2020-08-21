import autofit as af
import autolens as al

"""
In this pipeline, we'll perform a subhalo analysis which determines the attempts to detect subhalos by putting
subhalos at fixed intevals on a 2D (y,x) grid.

The mass model and source are initialized using an already run 'source' and 'mass' pipeline.

The pipeline is as follows:

Phase 1 - Lens Plane:

    Perform the subhalo detection analysis using a *GridSearch* of non-linear searches.

    Lens Mass: Previous mass pipeline model.
    Subhalo: SphericalNFWLudlow
    Source Light: Previous source pipeilne model.
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass (instance -> previous pipeline), source light (model -> previous pipeline).
    Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

Phase 1 - Foreground Plane:

    Perform the subhalo detection analysis.
    
    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Subhalo: SphericalNFWLudlow
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass (instance -> previous pipeline), source light (model -> previous pipeline).
    Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

Phase 2 - Background Plane:

    Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
    
    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Subhalo: SphericalNFWLudlow
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass & source light (model -> previous pipeline), subhalo mass (model -> phase 2).
    Notes: None

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
    This _GridPhase_ is used for all 3 subhalo detection phases, specifying that the subhalo (y,x) coordinates 
    are fitted for on a grid of non-linear searches.
    """

    class GridPhase(af.as_grid_search(phase_class=al.PhaseImaging, parallel=parallel)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    """
    Phase Lens Plane: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift is fixed to that of the lens galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        4) The lens mass model is fitted for simultaneously with the subhalo (it can be fixed if mass_as_model=False). 
        5) For an _Inversion_, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          _LightProfile_ they are varied (this is customized using source_as_model).
    """

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

    phase_lens_plane = GridPhase(
        phase_name="phase_1__subhalo_search__source__lens_plane",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=subhalo_search,
        number_of_steps=grid_size,
    )

    """
    Phase Foreground: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift has a UniformPrior between Earth and the lens galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        4) The lens mass model is fitted for simultaneously with the subhalo (it can be fixed if mass_as_model=False). 
        5) For an _Inversion_, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          _LightProfile_ they are varied (this is customized using source_as_model).
    """

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

    """
    Phase Background: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift has a UniformPrior between the lens galaxy and source galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        4) The lens mass model is fitted for simultaneously with the subhalo (it can be fixed if mass_as_model=False). 
        5) For an _Inversion_, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          _LightProfile_ they are varied (this is customized using source_as_model).
    """

    phase_foreground_plane = GridPhase(
        phase_name="phase_1__subhalo_search__source__foreground_plane",
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

    phase_background_plane = GridPhase(
        phase_name="phase_1__subhalo_search__source__background_plane",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo_z_above, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=subhalo_search,
        number_of_steps=5,
    )

    # subhalo = al.GalaxyModel(redshift=redshift_lens, mass=al.mp.SphericalNFWMCRLudlow)
    #
    # subhalo.mass.mass_at_200 = phase_lens_plane.result.model.galaxies.subhalo.mass.mass_at_200
    # subhalo.mass.centre = phase_lens_plane.result.model.galaxies.subhalo.mass.centre
    # subhalo.mass.redshift_object = redshift_lens
    #
    # source = slam.source_from_previous_pipeline_model_or_instance(
    #     source_as_model=True, index=-1
    # )
    #
    # subhalo.mass.redshift_source = redshift_source

    # phase2 = al.PhaseImaging(
    #     phase_name="phase_2__subhalo_refine",
    #     folders=folders,
    #     galaxies=dict(
    #         lens=af.last[-1].model.galaxies.lens, source=source, subhalo=subhalo
    #     ),
    #     hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
    #     hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
    #     settings=settings,
    #     search=af.DynestyStatic(n_live_points=100),
    # )
    #
    # phase2 = phase2.extend_with_multiple_hyper_phases(
    #     setup=slam.hyper, include_inversion=False
    # )

    return al.PipelineDataset(
        pipeline_name, phase_lens_plane, phase_background_plane, phase_foreground_plane
    )
