import autofit as af
import autolens as al

"""
In this pipeline, we fit `Imaging` of a strong lens system where:

 - The lens `Galaxy`'s `LightProfile` is modeled as an _EllipticalSersic_.
 - The lens `Galaxy`'s stellar `MassProfile` is fitted with the `EllipticalSersic` of the 
      `LightProfile`, where it is converted to a stellar mass distribution via a constant mass-to-light ratio.
 - The lens `Galaxy`'s dark matter `MassProfile` is modeled as a _SphericalNFW_.
 - The source `Galaxy`'s `LightProfile` is modeled as an _EllipticalSersic_.  

The pipeline is three phases:

Phase 1:

    Fit and subtract the lens light model.
    
    Lens Light: EllipticalSersic
    Lens Mass: None
    Source Light: None
    Prior Passing: None
    Notes: None

Phase 2:

    Fit the lens mass model and source `LightProfile`, where the `LightProfile` parameters of the lens`s 
    `LightMassProfile` are fixed to the results of phase 1.
    
    Lens Light: EllipticalSersic
    Lens Mass: EllipticalSersic + SphericalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens Light (instance -> phase 1).
    Notes: Uses the lens subtracted image from phase 1.

Phase 3:

    Refine the lens light and mass models and source light model using priors initialized from phases 1 and 2.
    
    Lens Light: EllipticalSersic
    Lens Mass: `EllipticalSersic` + SphericalNFWMCRLudlow + ExternalShear
    Source Light: EllipticalSersic
    Prior Passing: Lens light (model -> phase 1), lens mass and source light (model -> phase 2).
    Notes: None
"""


def make_pipeline(setup, settings):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    pipeline_name = "pipeline__light_sersic__mass_mlr_dark__source_sersic"

    """
    This pipeline is tagged according to whether:

      1) The lens galaxy mass model includes an  _ExternalShear_.
      2) The centres of the lens galaxy bulge and dark matter are aligned.
    """

    path_prefix = f"{setup.path_prefix}/{pipeline_name}/{setup.tag}"

    """
    Phase 1: Fit only the lens `Galaxy`'s light, where we:

        1) Set priors on the lens galaxy (y,x) centre such that we assume the image is centred around the lens galaxy.
    """

    sersic = af.PriorModel(al.lp.EllipticalSersic)
    sersic.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    sersic.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__light_sersic",
        path_prefix=path_prefix,
        galaxies=dict(lens=al.GalaxyModel(redshift=setup.redshift_lens, sersic=sersic)),
        settings=settings,
        search=af.DynestyStatic(n_live_points=50),
    )

    """
    Phase 2: Fit the lens`s `LightMassProfile` and `MassProfile` and source `Galaxy`'s light, where we:

        1) Fix the foreground lens light subtraction to the lens galaxy light model from phase 1.
        2) Set priors on the centre of the lens `Galaxy`'s dark matter `MassProfile` by linking them to those inferred 
           for the `LightProfile` in phase 1.
        3) Use a `SphericalNFWMCRLudlow` model for the dark matter which sets its scale radius via a mass-concentration
           relation and the lens and source redshifts.
    """

    sersic = af.PriorModel(al.lmp.EllipticalSersic)

    sersic.centre = phase1.result.instance.galaxies.lens.sersic.centre
    sersic.elliptical_comps = (
        phase1.result.instance.galaxies.lens.sersic.elliptical_comps
    )
    sersic.intensity = phase1.result.instance.galaxies.lens.sersic.intensity
    sersic.effective_radius = (
        phase1.result.instance.galaxies.lens.sersic.effective_radius
    )
    sersic.sersic_index = phase1.result.instance.galaxies.lens.sersic.sersic_index

    dark = af.PriorModel(al.mp.SphericalNFWMCRLudlow)

    """Setup: Align the centre of the `LightProfile` and dark matter `MassProfile` if input in _SetupMassLightDark_."""

    if setup.setup_mass.align_light_dark_centre:
        dark.centre = sersic.centre
    else:
        dark.centre = phase1.result.model.galaxies.lens.sersic.centre

    dark.mass_at_200 = af.LogUniformPrior(lower_limit=5e8, upper_limit=5e14)
    dark.redshift_object = setup.redshift_lens
    dark.setup.redshift_source = setup.redshift_source

    """Setup: Include an `ExternalShear` in the mass model if turned on in _SetupMass_. """

    if not setup.setup_mass.no_shear:
        shear = al.mp.ExternalShear
    else:
        shear = None

    phase2 = al.PhaseImaging(
        phase_name="phase_2__mass_mlr_dark__source_sersic__fixed_lens_light",
        path_prefix=path_prefix,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens, sersic=sersic, dark=dark, shear=shear
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source, sersic=al.lp.EllipticalSersic
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=60),
    )

    """
    Phase 3: Fit simultaneously the lens and source galaxies, where we:

        1) Set the lens`s light, mass, and source`s light using the results of phases 1 and 2.
    """

    sersic = af.PriorModel(al.lmp.EllipticalSersic)

    sersic.centre = phase1.result.model.galaxies.lens.sersic.centre
    sersic.elliptical_comps = phase1.result.model.galaxies.lens.sersic.elliptical_comps
    sersic.intensity = phase1.result.model.galaxies.lens.sersic.intensity
    sersic.effective_radius = phase1.result.model.galaxies.lens.sersic.effective_radius
    sersic.sersic_index = phase1.result.model.galaxies.lens.sersic.sersic_index
    sersic.mass_to_light_ratio = (
        phase2.result.model.galaxies.lens.sersic.mass_to_light_ratio
    )

    phase3 = al.PhaseImaging(
        phase_name="phase_3__light_sersic__mass_mlr_dark__source_sersic",
        path_prefix=path_prefix,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=setup.redshift_lens,
                sersic=sersic,
                dark=phase2.result.model.galaxies.lens.dark,
                shear=phase2.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=setup.redshift_source,
                sersic=phase2.result.model.galaxies.source.sersic,
            ),
        ),
        settings=settings,
        search=af.DynestyStatic(n_live_points=100),
    )

    return al.PipelineDataset(pipeline_name, phase1, phase2, phase3)
