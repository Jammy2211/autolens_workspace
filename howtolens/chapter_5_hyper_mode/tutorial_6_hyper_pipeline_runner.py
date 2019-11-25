import os

# In this tutorial, we'll go back to our complex source pipeline, but this time, as you've probably guessed, fit it
# using an inversion. As we discussed in tutorial 6, we'll begin by modeling the source with a light profile,
# to initialize the mass model, and then switch to an inversion.

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

# This function simulates the complex source, and is the same function we used in chapter 3, tutorial 3. It also adds
# lens galaxy light.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    grid = al.grid.uniform(shape_2d=(180, 180), pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.7,
            phi=80.0,
            intensity=0.8,
            effective_radius=1.3,
            sersic_index=2.5,
        ),
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


# Plot Imaging before running.
imaging = simulate()

# Remember, we need to define and pass our mask to the hyper_galaxies pipeline from the beginning.
mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

al.plot.imaging.subplot(imaging=imaging, mask=mask)

### HYPER PIPELINE SETTINGS ###

# Hopefully, you're used to seeing us use PipelineSettings to customize the behaviour of a pipeline. If not, they're fairly
# straight forward, they simmply allow us to choose how the pipeline behaves.

# Hyper-fitting brings with it the following settings:

# - If hyper_galaxies-galaxies are used to scale the noise in each component of the image (default True)

# - If the background sky is modeled throughout the pipeline (default False)

# - If the level of background noise is hyper throughout the pipeline (default True)

pipeline_settings = al.PipelineSettingsHyper(
    hyper_galaxies=True,
    hyper_image_sky=False,
    hyper_background_noise=False,
    include_shear=True,
    pixelization=al.pix.VoronoiBrightnessImage,
    regularization=al.reg.AdaptiveBrightness,
)

# Lets import the pipeline and run it.
from howtolens.chapter_5_hyper_mode import tutorial_6_hyper_pipeline

pipeline_hyper = tutorial_6_hyper_pipeline.make_pipeline(
    pipeline_settings=pipeline_settings, phase_folders=["howtolens", "c5_t6_hyper"]
)

pipeline_hyper.run(dataset=imaging, mask=mask)
