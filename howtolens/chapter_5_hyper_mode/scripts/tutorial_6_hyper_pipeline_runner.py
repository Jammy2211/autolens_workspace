# %%
"""
In this tutorial, we'll go back to our complex source pipeline, but this time, as you've probably guessed, fit it
using an inversion. As we discussed in tutorial 6, we'll begin by modeling the source with a light profile,
to initialize the mass model, and then switch to an inversion.
"""

# %%
import autofit as af

# %%
"""
Setup the path to the autolens_workspace, using a relative directory name.
"""

# %%
workspace_path = "/path/to/user/autolens_workspace/"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace/"

# %%
"""
Setup the path to the config folder, using the autolens_workspace path.
"""

# %%
config_path = workspace_path + "config"

# %%
"""
Use this path to explicitly set the config path and output path.
"""

# %%
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

import autolens as al
import autolens.plot as aplt

# %%
"""
This function simulates the complex source, and is the same function we used in chapter 3, tutorial 3. It also adds
lens galaxy light.
"""

# %%
def simulate():

    grid = al.Grid.uniform(shape_2d=(150, 150), pixel_scales=0.05, sub_size=2)

    psf = al.Kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

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

    simulator = al.SimulatorImaging(
        exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        psf=psf,
        background_sky_map=al.Array.full(fill_value=1.0, shape_2d=grid.shape_2d),
        add_noise=True,
        noise_seed=1,
    )

    return simulator.from_tracer_and_grid(tracer=tracer, grid=grid)


# %%
"""
Plot Imaging before running.
"""

# %%
imaging = simulate()

# %%
"""
Remember, we need to define and pass our mask to the hyper_galaxies pipeline from the beginning.
"""

# %%
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
The setup module customizes the behaviour of a pipeline. Hyper-fitting brings with it the following setup:

- If hyper-galaxies are used to scale the noise in each component of the image (default True)
- If the level of background noise is modeled throughout the pipeline (default True)
- If the background sky is modeled throughout the pipeline (default False)
"""

# %%
general_setup = al.setup.General(
    hyper_galaxies=True,
    hyper_background_noise=True,
    hyper_image_sky=False,  # <- By default this feature is off, as it rarely changes the lens model.
)

# %%
"""
Source setup are required for the inversion. With hyper-mode on we can now use the VoronoiBrightnessImage
and AdaptiveBrightness classes which adapt to the source's surface-brightness.
"""

# %%
source_setup = al.setup.Source(
    pixelization=al.pix.VoronoiBrightnessImage, regularization=al.reg.AdaptiveBrightness
)

setup = al.setup.Setup(general=general_setup, source=source_setup)

# %%
"""
Lets import the pipeline and run it.
"""

# %%
from howtolens.chapter_5_hyper_mode import tutorial_6_hyper_pipeline

pipeline_hyper = tutorial_6_hyper_pipeline.make_pipeline(
    setup=setup, phase_folders=["howtolens", "c5_t6_hyper"]
)

# pipeline_hyper.run(dataset=imaging, mask=mask)
