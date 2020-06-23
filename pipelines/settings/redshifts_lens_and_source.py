import autofit as af
import autolens as al

"""
In this pipeline, we'll demonstrate passing redshifts to a pipeline - which means that the results and images of this
pipeline will be returned in physical unit_label (e.g. lengths in kpcs as well as arcsec, luminosities in magnitudes,
masses in solMass, etc).

The redshift of the lens and source are input parameters of all pipelines, and they take default values of 0.5 and
1.0. Thus, *all* pipelines will return physical values assuming these fiducial values if no other values are
specified. Care must be taken interpreting the distances and masses if these redshifts are not correct or if the
true redshifts of the lens and / or source galaxies are unknown.

We'll perform a basic analysis which fits a lensed source galaxy using a parametric _LightProfile_ where
the lens's light is omitted. This pipeline uses two phases:

Phase 1:

Description: Fit the lens mass model and source _LightProfile_ using x1 source.
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: EllipticalSersic
Prior Passing: None
Notes: Inputs the pipeline default redshifts where the lens has redshift 0.5, source 1.0.

Phase 1:

Description: Fit the lens and source model again..
Lens Mass: EllipticalIsothermal + ExternalShear
Source Light: EllipticalSersic
Prior Passing: Lens mass (model -> phase 1), source light (model -> phase 1)
Notes: Manually over-rides the lens redshift to 1.0 and source redshift to 2.0, to illustrate the different results.
"""


def make_pipeline(settings, folders=None, redshift_lens=0.5, redshift_source=1.0):

    """SETUP PIPELINE & PHASE NAMES, TAGS AND PATHS"""

    # We setup the pipeline name using the tagging module. In this case, the pipeline name is not given a tag and
    # will be the string specified below. However, its good practise to use the 'tag.' function below, incase
    # a pipeline does use customized tag names.

    pipeline_name = "pipeline__settings__redshifts"

    # Unlike other features, the redshifts of the lens and source do not change the setup tag and phase path. Thus,
    # our output will simply go to the phase path:

    # phase_path = 'phase_name/setup'

    # This function uses the phase folders and pipeline name to set up the output directory structure,
    # e.g. 'autolens_workspace/output/pipeline_name/pipeline_tag/phase_name/phase_tag//'

    setup.folders.append(pipeline_name)

    ### Phase 1 ###

    # In phase 1, we fit the lens galaxy's mass and one source galaxy, where we:

    # 1) Use the input value of redshifts from the pipeline.

    mass = af.PriorModel(al.mp.EllipticalIsothermal)
    mass.centre_0 = af.GaussianPrior(mean=0.0, sigma=0.1)
    mass.centre_1 = af.GaussianPrior(mean=0.0, sigma=0.1)

    phase1 = al.PhaseImaging(
        phase_name="phase_1__x1_source",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=redshift_lens, mass=mass, shear=al.mp.ExternalShear
            ),
            source_0=al.GalaxyModel(
                redshift=redshift_source, light=al.lp.EllipticalSersic
            ),
        ),
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 80
    phase1.search.facc = 0.2

    ### Phase 2 ###

    # In phase 2, we fit the lens galaxy's mass and two source galaxies, where we:

    # 1) Use manually specified new values of redshifts for the lens and source galaxies.

    phase2 = al.PhaseImaging(
        phase_name="phase_2__x2_source",
        folders=setup.folders,
        galaxies=dict(
            lens=al.GalaxyModel(
                redshift=1.0,
                mass=phase1.result.model.galaxies.lens.mass,
                shear=phase1.result.model.galaxies.lens.shear,
            ),
            source=al.GalaxyModel(
                redshift=2.0, light=phase1.result.model.galaxies.source.light
            ),
        ),
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 50
    phase2.search.facc = 0.3

    return al.PipelineDataset(pipeline_name, phase1, phase2)
