# %%
"""
Up to now, we've not paid much attention to the source galaxy's morphology. We've assumed its a single-component
exponential profile, which is a fairly crude assumption. A quick look at any image of a real galaxy reveals a
wealth of different structures that could be present - bulges, disks, bars, star-forming knots and so on. Furthermore,
there could be more than one source-galaxy!

In this example, we'll explore how far we can get trying to_fit a complex source using a pipeline. Fitting complex
source's is an exercise in diminishing returns. Each component we add to our source model brings with it an
extra 5-7, parameters. If there are 4 components, or multiple galaxies, we're quickly entering the somewhat nasty
regime of 30-40+ parameters in our non-linear search. Even with a pipeline, that is a lot of parameters to fit!
"""

# %%
""" AUTOFIT + CONFIG SETUP """

import autofit as af

# %%
"""
Setup the path to the workspace, using by filling in your path below.
"""

# %%
workspace_path = "/path/to/user/autolens_workspace/"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"

# %%
"""
Use this path to explicitly set the config path and output path.
"""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
#%matplotlib inline

import autolens as al
import autolens.plot as aplt

# %%
"""
This function simulates an image with a complex source.
"""

# %%
def simulate():

    grid = al.Grid.uniform(shape_2d=(180, 180), pixel_scales=0.05, sub_size=1)

    psf = al.Kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.1, 0.0), einstein_radius=1.6
        ),
    )

    source_galaxy_0 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.1, 0.1),
            elliptical_comps=(0.1, 0.0),
            intensity=0.2,
            effective_radius=1.0,
            sersic_index=1.5,
        ),
    )

    source_galaxy_1 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(-0.25, 0.25),
            elliptical_comps=(0.0, 0.15),
            intensity=0.1,
            effective_radius=0.2,
            sersic_index=3.0,
        ),
    )

    source_galaxy_2 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.45, -0.35),
            elliptical_comps=(0.0, 0.222222),
            intensity=0.03,
            effective_radius=0.3,
            sersic_index=3.5,
        ),
    )

    source_galaxy_3 = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(-0.05, -0.0),
            elliptical_comps=(0.05, 0.1),
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
        background_sky_map=al.Array.full(fill_value=0.1, shape_2d=grid.shape_2d),
        add_noise=True,
    )

    return simulator.from_tracer_and_grid(tracer=tracer, grid=grid)


# %%
"""
Lets Simulate the imaging data and set up the mask.
"""

# %%
imaging = simulate()

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
Yep, that's a pretty complex source. There are clearly more than 4 peaks of light - I wouldn't like to guess whether 
there is one or two sources (or more). You'll also notice I omitted the lens galaxy's light for this system. This is to 
keep the number of parameters down and the phases running fast, but we wouldn't get such a luxury for a real galaxy.

Again, before we checkout the pipeline, lets import it, and get it running.
"""

# %%
from howtolens.chapter_3_pipelines import tutorial_3_pipeline_complex_source

pipeline_complex_source = tutorial_3_pipeline_complex_source.make_pipeline(
    phase_folders=["howtolens", "c3_t3_complex_source"]
)

# pipeline_complex_source.run(dataset=imaging, mask=mask)

# %%
"""
Okay, so with 4 sources, we still couldn't get a good a fit to the source that didn't leave residuals. However, I 
actually simulated the lens with 4 sources. This means that there is a 'perfect fit' somewhere in parameter space 
that we unfortunately missed using the pipeline above.

Lets confirm this, by manually fitting the imaging data with the true input model.
"""

# %%
masked_imaging = al.MaskedImaging(
    imaging=imaging,
    mask=al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    ),
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.1, 0.0), einstein_radius=1.6
    ),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=(0.0, 0.1),
        intensity=0.2,
        effective_radius=1.0,
        sersic_index=1.5,
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(-0.25, 0.25),
        elliptical_comps=(0.0, 0.15),
        intensity=0.1,
        effective_radius=0.2,
        sersic_index=3.0,
    ),
)

source_galaxy_2 = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.45, -0.35),
        elliptical_comps=(0.0, 0.222222),
        intensity=0.03,
        effective_radius=0.3,
        sersic_index=3.5,
    ),
)

source_galaxy_3 = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(-0.05, -0.0),
        elliptical_comps=(0.0, 0.15),
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

true_fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

aplt.FitImaging.subplot_fit_imaging(fit=true_fit)

aplt.FitImaging.subplot_of_plane(fit=true_fit, plane_index=1)


# %%
"""
And indeed, we see an improved residual-map, chi-squared-map, and so forth.

The morale of this story is that if the source morphology is complex, there is no way we can build a pipeline to 
fit it. For this tutorial, this was true even though our source model could actually fit the data perfectly. For real 
lenses, the source will be *even more complex* and there is even less hope of getting a good fit :(

But fear not, PyAutoLens has you covered. In chapter 4, we'll introduce a completely new way to model the source 
galaxy, which addresses the problem faced here. But before that, in the next tutorial we'll discuss how we actually 
pass priors in a pipeline.
"""
