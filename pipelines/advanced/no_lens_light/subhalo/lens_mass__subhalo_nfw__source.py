import autofit as af
import autolens as al

# In this pipeline, we'll perform a subhalo analysis which determines the attempts to detect subhalos by putting
# subhalos at fixed intevals on a 2D (y,x) grid.

# The mass model and source are initialized using an already run 'source' and 'mass' pipeline.

# The pipeline is as follows:

# Phase 1:

# Perform the subhalo detection analysis.

# Lens Mass: Previous mass pipeline model.
# Source Light: Previous source pipeilne model.
# Subhalo: SphericalNFWLudlow
# Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
# Prior Passing: Lens mass (instance -> previous pipeline), source light (model -> previous pipeline).
# Notes: Priors on subhalo are tuned to give realistic masses (10^6 - 10^10) and concentrations (6-24)

# Phase 2:

# Refine the best-fit detected subhalo from the previous phase, by varying also the lens mass model.

# Lens Mass: Previous mass pipeline model.
# Source Light: Previous source pipeilne model.
# Subhalo: SphericalNFWLudlow
# Previous Pipeline: no_lens_light/mass/*/lens_*__source.py
# Prior Passing: Lens mass & source light (model -> previous pipeline), subhalo mass (model -> phase 2).
# Notes: None


def source_with_previous_model_or_instance(setup, source_as_model=False, index=0):
    """Setup the source source model using the previous pipeline or phase results.

    This function is required because the source light model is not specified by the pipeline itself (e.g. the previous
    pipelines determines if the source was modeled using parametric light profiles or an inversion.

    If the source was parametric this function returns the source as a model, given that a parametric source should be
    fitted for simultaneously with the mass model.

    If the source was an inversion then it is returned as an instance, given that the inversion parameters do not need
    to be fitted for alongside the mass model.

    The bool include_hyper_source determines if the hyper-galaxy used to scale the sources noises is included in the
    model fitting.
    """

    if setup.general.hyper_galaxies:

        hyper_galaxy = af.PriorModel(al.HyperGalaxy)

        hyper_galaxy.noise_factor = (
            af.last.hyper_combined.model.galaxies.source.hyper_galaxy.noise_factor
        )
        hyper_galaxy.contribution_factor = (
            af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.contribution_factor
        )
        hyper_galaxy.noise_power = (
            af.last.hyper_combined.instance.optional.galaxies.source.hyper_galaxy.noise_power
        )

    else:

        hyper_galaxy = None

    if setup.source.type_tag in "sersic":

        if source_as_model:

            return al.GalaxyModel(
                redshift=af.last[index].model.galaxies.source.redshift,
                sersic=af.last[index].model.galaxies.source.sersic,
                hyper_galaxy=hyper_galaxy,
            )

        else:

            return al.GalaxyModel(
                redshift=af.last[index].instance.galaxies.source.redshift,
                sersic=af.last[index].instance.galaxies.source.sersic,
                hyper_galaxy=hyper_galaxy,
            )

    else:

        if source_as_model:

            return al.GalaxyModel(
                redshift=af.last.instance.galaxies.source.redshift,
                pixelization=af.last.hyper_combined.instance.galaxies.source.pixelization,
                regularization=af.last.hyper_combined.model.galaxies.source.regularization,
            )

        else:

            return al.GalaxyModel(
                redshift=af.last.instance.galaxies.source.redshift,
                pixelization=af.last[
                    index
                ].hyper_combined.instance.galaxies.source.pixelization,
                regularization=af.last[
                    index
                ].hyper_combined.instance.galaxies.source.regularization,
                hyper_galaxy=hyper_galaxy,
            )


def make_pipeline(
    setup,
    phase_folders=None,
    redshift_lens=0.5,
    redshift_source=1.0,
    auto_positions_factor=None,
    positions_threshold=None,
    sub_size=2,
    signal_to_noise_limit=None,
    bin_up_factor=None,
    pixel_scale_interpolation_grid=None,
    inversion_uses_border=True,
    inversion_pixel_limit=None,
    parallel=False,
):

    ### SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS ###

    pipeline_name = "pipeline_subhalo__nfw"

    # This pipeline is tagged according to whether:

    # 1) Hyper-fitting setup (galaxies, sky, background noise) are used.
    # 2) The lens galaxy mass model includes an external shear.

    phase_folders.append(pipeline_name)
    phase_folders.append(setup.general.tag)
    phase_folders.append(setup.source.tag)
    phase_folders.append(setup.mass.tag)

    ### Phase 1 ###

    # In phase 1, we attempt to detect subhalos, by performing a NxN grid search of MultiNest searches, where:

    # 1) The lens model and source parameters are held fixed to the best-fit values of the previous pipeline.
    # 2) Each grid search varies the subhalo (y,x) coordinates and mass as free parameters.
    # 3) The priors on these (y,x) coordinates are UniformPriors, with limits corresponding to the grid-cells.

    class GridPhase(af.as_grid_search(phase_class=al.PhaseImaging, parallel=parallel)):
        @property
        def grid_priors(self):
            return [
                self.model.galaxies.subhalo.mass.centre_0,
                self.model.galaxies.subhalo.mass.centre_1,
            ]

    subhalo = al.GalaxyModel(redshift=redshift_lens, mass=al.mp.SphericalNFWMCRLudlow)

    subhalo.mass.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e6, upper_limit=1.0e11)
    subhalo.mass.centre_0 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)
    subhalo.mass.centre_1 = af.UniformPrior(lower_limit=-2.0, upper_limit=2.0)

    subhalo.mass.redshift_object = subhalo.redshift

    # Setup the source model, which uses a variable parametric profile or fixed inversion model depending on the
    # previous pipeline.

    source = source_with_previous_model_or_instance(
        setup=setup, source_as_model=False, index=0
    )

    subhalo.mass.redshift_source = source.redshift

    phase1 = GridPhase(
        phase_name="phase_1__subhalo_search__source",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=af.last.instance.galaxies.lens, subhalo=subhalo, source=source
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        auto_positions_factor=auto_positions_factor,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        non_linear_class=af.MultiNest,
        number_of_steps=5,
    )

    phase1.optimizer.const_efficiency_mode = False
    phase1.optimizer.n_live_points = 50
    phase1.optimizer.sampling_efficiency = 0.2
    phase1.optimizer.evidence_tolerance = 3.0

    subhalo = al.GalaxyModel(redshift=redshift_lens, mass=al.mp.SphericalNFWMCRLudlow)

    subhalo.mass.mass_at_200 = phase1.result.model.galaxies.subhalo.mass.mass_at_200
    subhalo.mass.centre = phase1.result.model_absolute(
        a=0.5
    ).galaxies.subhalo.mass.centre

    source = source_with_previous_model_or_instance(
        setup=setup, source_as_model=True, index=-1
    )

    phase2 = al.PhaseImaging(
        phase_name="phase_2__subhalo_refine",
        phase_folders=phase_folders,
        galaxies=dict(
            lens=af.last[-1].model.galaxies.lens, subhalo=subhalo, source=source
        ),
        hyper_image_sky=af.last.hyper_combined.instance.optional.hyper_image_sky,
        hyper_background_noise=af.last.hyper_combined.instance.optional.hyper_background_noise,
        positions_threshold=positions_threshold,
        auto_positions_factor=auto_positions_factor,
        sub_size=sub_size,
        signal_to_noise_limit=signal_to_noise_limit,
        bin_up_factor=bin_up_factor,
        pixel_scale_interpolation_grid=pixel_scale_interpolation_grid,
        inversion_uses_border=inversion_uses_border,
        inversion_pixel_limit=inversion_pixel_limit,
        non_linear_class=af.MultiNest,
    )

    phase2.optimizer.const_efficiency_mode = False
    phase2.optimizer.n_live_points = 80
    phase2.optimizer.sampling_efficiency = 0.3
    phase2.optimizer.evidence_tolerance = 0.8

    phase2 = phase2.extend_with_multiple_hyper_phases(
        hyper_galaxy=setup.general.hyper_galaxies,
        include_background_sky=setup.general.hyper_image_sky,
        include_background_noise=setup.general.hyper_background_noise,
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2)
