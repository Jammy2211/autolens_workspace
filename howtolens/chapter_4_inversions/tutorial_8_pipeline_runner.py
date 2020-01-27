# In this tutorial, we'll go back to our complex source pipeline, but this time, as you've probably guessed, fit it
# using an inversion. As we discussed in tutorial 6, we'll begin by modeling the source with a light profile,
# to initialize the mass model, and then switch to an inversion.

import os

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al
import autolens.plot as aplt

# This function simulates the complex source, and is the same function we used in chapter 3, tutorial 3.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    grid = al.grid.uniform(shape_2d=(180, 180), pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
        ),
    )

    source_galaxy_0 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.1, 0.1),
            axis_ratio=0.8,
            phi=90.0,
            intensity=0.2,
            effective_radius=1.0,
            sersic_index=1.5,
        ),
    )

    source_galaxy_1 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(-0.25, 0.25),
            axis_ratio=0.7,
            phi=45.0,
            intensity=0.1,
            effective_radius=0.2,
            sersic_index=3.0,
        ),
    )

    source_galaxy_2 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.45, -0.35),
            axis_ratio=0.6,
            phi=90.0,
            intensity=0.03,
            effective_radius=0.3,
            sersic_index=3.5,
        ),
    )

    source_galaxy_3 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(-0.05, -0.0),
            axis_ratio=0.9,
            phi=140.0,
            intensity=0.03,
            effective_radius=0.1,
            sersic_index=4.0,
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[
            lens_galaxy,
            source_galaxy_0,
            source_galaxy_1,
            source_galaxy_2,
            source_galaxy_3,
        ]
    )

    return al.SimulatedImagingData.from_tracer(
        tracer=tracer,
        grid=grid,
        pixel_scales=0.05,
        exposure_time=300.0,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )


# Lets simulate the we'll fit, which is the same complex source as the
# 'chapter_3_pipelines/tutorial_3_complex_source.py' tutorial.
imaging = simulate()
aplt.imaging.subplot_imaging(imaging=imaging)

# Lets import the pipeline and run it.
from howtolens.chapter_4_inversions import tutorial_8_pipeline

pipeline_inversion = tutorial_8_pipeline.make_pipeline(
    phase_folders=["howtolens", "c4_t8_inversion"]
)
pipeline_inversion.run(dataset=imaging)

# And with that, we now have a pipeline to model strong lenses using an inversion! Checkout the example pipeline in
# 'autolens_workspace/pipelines/examples/inversion_hyper_galaxies_bg_noise.py' for an example of an inversion pipeline that includes the lens light
# component.
