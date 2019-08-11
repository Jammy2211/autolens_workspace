import autofit as af
from autolens.data.array import mask as msk
from autolens.data.instrument import abstract_data
from autolens.data.instrument import ccd
from autolens.data.plotters import ccd_plotters
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.pipeline import pipeline as pl
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion import regularization as reg

import os

# In this tutorial, we'll go back to our complex source pipeline, but this time, as you've probably guessed, fit it
# using an inversion. As we discussed in tutorial 6, we'll begin by modeling the source with a light profile,
# to initialize the mass model, and then switch to an inversion.

# To setup the config and output paths without docker, you need to uncomment and run the command below.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

# This function simulates the complex source, and is the same function we used in chapter 3, tutorial 3. It also adds
# lens galaxy light.
def simulate():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = abstract_data.PSF.from_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=0.05)

    image_plane_grid_stack = grids.GridStack.from_shape_pixel_scale_and_sub_grid_size(
        shape=(180, 180), pixel_scale=0.05, psf_shape=(11, 11)
    )

    lens_galaxy = g.Galaxy(
        redshift=0.5,
        light=lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.7,
            phi=80.0,
            intensity=0.8,
            effective_radius=1.3,
            sersic_index=2.5,
        ),
        mass=mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=135.0, einstein_radius=1.6
        ),
    )

    source_galaxy_0 = g.Galaxy(
        redshift=1.0,
        light=lp.EllipticalSersic(
            centre=(0.1, 0.1),
            axis_ratio=0.8,
            phi=90.0,
            intensity=0.2,
            effective_radius=1.0,
            sersic_index=1.5,
        ),
    )

    source_galaxy_1 = g.Galaxy(
        redshift=1.0,
        light=lp.EllipticalSersic(
            centre=(-0.25, 0.25),
            axis_ratio=0.7,
            phi=45.0,
            intensity=0.1,
            effective_radius=0.2,
            sersic_index=3.0,
        ),
    )

    source_galaxy_2 = g.Galaxy(
        redshift=1.0,
        light=lp.EllipticalSersic(
            centre=(0.45, -0.35),
            axis_ratio=0.6,
            phi=90.0,
            intensity=0.03,
            effective_radius=0.3,
            sersic_index=3.5,
        ),
    )

    source_galaxy_3 = g.Galaxy(
        redshift=1.0,
        light=lp.EllipticalSersic(
            centre=(-0.05, -0.0),
            axis_ratio=0.9,
            phi=140.0,
            intensity=0.03,
            effective_radius=0.1,
            sersic_index=4.0,
        ),
    )

    tracer = ray_tracing.Tracer.from_galaxies_and_image_plane_grid_stack(
        galaxies=[
            lens_galaxy,
            source_galaxy_0,
            source_galaxy_1,
            source_galaxy_2,
            source_galaxy_3,
        ],
        image_plane_grid_stack=image_plane_grid_stack,
    )

    return ccd.SimulatedCCDData.from_tracer_and_exposure_arrays(
        tracer=tracer,
        pixel_scale=0.05,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


# Plot CCD before running.
ccd_data = simulate()
ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# Remember, we need to define and pass our mask to the hyper_galaxy pipeline from the beginning.
mask = msk.Mask.circular(
    shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0
)

### HYPER PIPELINE SETTINGS ###

# Hopefully, you're used to seeing us use PipelineSettings to customize the behaviour of a pipeline. If not, they're fairly
# straight forward, they simmply allow us to choose how the pipeline behaves.

# Hyper-fitting brings with it the following settings:

# - If hyper_galaxy-galaxies are used to scale the noise in each component of the image (default True)

# - If the background sky is modeled throughout the pipeline (default False)

# - If the level of background noise is normal throughout the pipeline (default True)

pipeline_settings = pl.PipelineSettingsHyper(
    hyper_galaxies=True,
    hyper_image_sky=False,
    hyper_background_noise=False,
    include_shear=True,
    pixelization=pix.VoronoiBrightnessImage,
    regularization=reg.AdaptiveBrightness,
)

# Lets import the pipeline and run it.
from workspace.howtolens.chapter_5_hyper_mode import tutorial_6_hyper_pipeline

pipeline_hyper = tutorial_6_hyper_pipeline.make_pipeline(
    pipeline_settings=pipeline_settings, phase_folders=["howtolens", "c5_t4_hyper"]
)

pipeline_hyper.run(data=ccd_data)
