import autofit as af
import autolens as al

"""
This pipeline performs a subhalo analysis which determines the attempts to detect subhalos by putting
subhalos at fixed intevals on a 2D (y,x) grid.

The mass model and source are initialized using an already run `source` and `mass` pipeline.

The pipeline is as follows:

Phase 1:

    Refine the lens mass model using the hyper-parameters optimized in the mass pipeline. This model should be used
    as the no substructure comparison model to quantify a substructure detection`s evidence increase.

    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass (model -> previous pipeline), source light (instance or model -> previous pipeline).

Phase 2 (part 1) - Lens Plane:

    Perform the subhalo detection analysis using a *GridSearch* of non-linear searches.

    Lens Mass: Previous mass pipeline model.
    Subhalo: SphericalNFWLudlow
    Source Light: Previous source pipeilne model.
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass (phase 1), source light (instance or model -> previous pipeline).
    Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^11).

Phase 2 (part 2) - Foreground Plane:

    Perform the subhalo detection analysis.
    
    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Subhalo: SphericalNFWLudlow
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass (phase 1), source light (instance or model -> previous pipeline).
    Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^11).

Phase 2 (part 3) - Background Plane:

    Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.
    
    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Subhalo: SphericalNFWLudlow
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass & source light (model ->phase 1), subhalo mass (instance or model -> phase 2).
    Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^11).
    
Phase 3:

Refine the best-fit detected subhalo from the previous phase.

    Lens Mass: Previous mass pipeline model.
    Source Light: Previous source pipeilne model.
    Subhalo: SphericalNFWLudlow
    Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
    Prior Passing: Lens mass & source light (model -> previous pipeline), subhalo mass (model -> phase 2).
    Notes: None
"""


def make_pipeline(slam, settings):
    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline_subhalo"

    """
    This pipeline is tagged according to whether:

        1) Hyper-fitting settings (galaxies, sky, background noise) are used.
        2) The lens galaxy mass model includes an  `ExternalShear`.
    """

    folders = slam.folders + [
        pipeline_name,
        slam.source_tag,
        slam.mass_tag,
        slam.setup_subhalo.tag,
    ]

    """
    Phase1 : Refit the lens`s `MassProfile``s and source, where we:

        1) Use the source galaxy model of the `source` pipeline.
        2) Fit this source as a model if it is parametric and as an instance if it is an `Inversion`.
    """

    """SLaM: Setup the lens and source passing them from the previous pipelines in the same way as described above."""

    lens = slam.lens_for_subhalo_pipeline()
    source = slam.source_from_previous_pipeline_model_if_parametric()

    phase1 = al.PhaseImaging(
        phase_name="phase_1__refine_mass",
        folders=folders,
        galaxies=dict(lens=lens, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    """
    This `GridPhase` is used for all 3 subhalo detection phases, specifying that the subhalo (y,x) coordinates 
    are fitted for on a grid of non-linear searches.
    """

    class GridPhase(
        af.as_grid_search(
            phase_class=al.PhaseImaging, parallel=slam.setup_subhalo.parallel
        )
    ):
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
        4) The lens mass model is fitted for simultaneously with the subhalo (it can be fixed if mass_is_model=False). 
        5) For an `Inversion`, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          `LightProfile` they are varied (this is customized using source_is_model).
    """

    subhalo = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)

    subhalo.mass.redshift_object = slam.redshift_lens
    subhalo.mass.redshift_source = slam.redshift_source

    """
    SLaM: Setup the lens model, which uses the phase1 result and is a model or instance depending on the
          *mass_is_model* parameter of `SetupSubhalo`.
    """

    lens = slam.lens_for_subhalo_pipeline()

    """
    SLaM: Setup the source model, which uses the the phase1 result is a model or instance depending on the 
    *source_is_model* parameter of `SetupSubhalo`.
    """

    source = slam.source_for_subhalo_pipeline()

    phase2_lens_plane = GridPhase(
        phase_name="phase_2__subhalo_search__lens_plane",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=slam.setup_subhalo.subhalo_search,
        number_of_steps=slam.setup_subhalo.grid_size,
    )

    """
    Phase Foreground: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift has a UniformPrior between Earth and the lens galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        4) The lens mass model is fitted for simultaneously with the subhalo (it can be fixed if mass_is_model=False). 
        5) For an `Inversion`, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          `LightProfile` they are varied (this is customized using source_is_model).
    """

    """The subhalo redshift is free to vary between 0.0 and the lens galaxy redshift."""

    subhalo_z_below = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo_z_below.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo_z_below.mass.centre_0 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_below.mass.centre_1 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_below.mass.redshift_source = slam.redshift_source
    subhalo_z_below.mass.redshift_object = af.UniformPrior(
        lower_limit=0.0, upper_limit=slam.redshift_lens
    )

    """
    Phase Background: attempt to detect subhalos, by performing a NxN grid search of non-linear searches, where:

        1) The subhalo redshift has a UniformPrior between the lens galaxy and source galaxy.
        2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
        3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.
        4) The lens mass model is fitted for simultaneously with the subhalo (it can be fixed if mass_is_model=False). 
        5) For an `Inversion`, the source parameters are fixed to the best-fit values of the previous pipeline, for a 
          `LightProfile` they are varied (this is customized using source_is_model).
    """

    phase2_foreground_plane = GridPhase(
        phase_name="phase_2__subhalo_search__foreground_plane",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo_z_below, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=slam.setup_subhalo.subhalo_search,
        number_of_steps=slam.setup_subhalo.grid_size,
    )

    """The subhalo redshift is free to vary between and the lens and source galaxy redshifts."""

    subhalo_z_above = al.GalaxyModel(
        redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow
    )

    subhalo_z_above.mass.mass_at_200 = af.LogUniformPrior(
        lower_limit=1.0e6, upper_limit=1.0e11
    )
    subhalo_z_above.mass.centre_0 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_above.mass.centre_1 = af.UniformPrior(lower_limit=-2.5, upper_limit=2.5)
    subhalo_z_above.mass.redshift_source = slam.redshift_source
    subhalo_z_above.mass.redshift_object = af.UniformPrior(
        lower_limit=slam.redshift_lens, upper_limit=slam.redshift_source
    )

    phase2_background_plane = GridPhase(
        phase_name="phase_2__subhalo_search__background_plane",
        folders=folders,
        galaxies=dict(lens=lens, subhalo=subhalo_z_above, source=source),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        settings=settings,
        search=slam.setup_subhalo.subhalo_search,
        number_of_steps=slam.setup_subhalo.grid_size,
    )

    # subhalo = al.GalaxyModel(redshift=slam.redshift_lens, mass=al.mp.SphericalNFWMCRLudlow)
    #
    # subhalo.mass.mass_at_200 = phase_lens_plane.result.model.galaxies.subhalo.mass.mass_at_200
    # subhalo.mass.centre = phase_lens_plane.result.model.galaxies.subhalo.mass.centre
    # subhalo.mass.redshift_object = slam.redshift_lens
    #
    # source = slam.source_from_previous_pipeline_model_or_instance(
    #     source_is_model=True, index=-1
    # )
    #
    # subhalo.mass.redshift_source = slam.redshift_source

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
        pipeline_name,
        phase1,
        phase2_lens_plane,
        phase2_foreground_plane,
        phase2_background_plane,
    )
