# Up to now, all the images we've fitted had one lens galaxy. However, we saw in chapter 1 that our lens plane can consist
# of multiple galaxies which each contribute to the strong lens. Multi-galaxy systems are challenging to
# model, because they add an extra 5-10 parameters to the non-linear search and, more problematically, the
# degeneracies between the mass-profiles of the two galaxies can be severe.

# However, we can still break their analysis down using a pipeline and give ourselves a shot at getting a good
# lens model. Here, we're going to fit a double lens system, fitting as much about each individual lens galaxy before
# fitting them simultaneously.

# Up to now, I've put a focus on pipelines being general. The pipeline we write in this example is going to be
# the opposite - specific to the image we're modeling. Fitting multiple lens galaxies is really difficult and
# writing a pipeline that we can generalize to many lenses isn't currently possible with PyAutoLens.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using by filling in your path below.
workspace_path = "/path/to/user/autolens_workspace/"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace/"

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=config_path, output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al
import autolens.plot as aplt

# This rather long simulator function generates an image with two strong lens galaxies.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy_0 = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(
            centre=(0.0, -1.0),
            axis_ratio=0.8,
            phi=55.0,
            intensity=0.1,
            effective_radius=0.8,
            sersic_index=2.5,
        ),
        mass=al.mp.EllipticalIsothermal(
            centre=(1.0, 0.0), axis_ratio=0.7, phi=45.0, einstein_radius=1.0
        ),
    )

    lens_galaxy_1 = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 1.0),
            axis_ratio=0.8,
            phi=100.0,
            intensity=0.1,
            effective_radius=0.6,
            sersic_index=3.0,
        ),
        mass=al.mp.EllipticalIsothermal(
            centre=(-1.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=0.8
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalExponential(
            centre=(0.05, 0.15), intensity=0.2, effective_radius=0.5
        ),
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy]
    )

    simulator = al.simulator.imaging(
        shape_2d=(180, 180),
        pixel_scales=0.05,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


# Lets Simulate the imaging datawe'll fit, which is a new image, finally!
imaging = simulate()

# We need to choose our mask for the analysis. Given the lens light is present in the image we'll need to include
# all of its light in the central regions of the image, so lets use a circular mask.
mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)

# Looking at the image, there are clearly two blobs of light corresponding to our two lens galaxies. The source's light
# is also pretty complex - the arcs don't posses the rotational symmetry we're used to seeing up to now. Multi-galaxy
# ray-tracing is just a lot more complicated, which means so is modeling it!

# So, how can we break the lens modeling up? As follows:

# 1) Fit and subtract the light of each lens galaxy individually - this will require some careful masking but is doable.
# 2) Use these results to initialize each lens galaxy's mass-profile.

# So, with this in mind, we've written a pipeline composed of 4 phases:

# Phase 1) Fit the light profile of the lens galaxy on the left of the image, at coordinates (0.0", -1.0").
# Phase 2) Fit the light profile of the lens galaxy on the right of the image, at coordinates (0.0", 1.0").
# Phase 3) Use this lens-subtracted image to fit the source galaxy's light. The mass-profiles of the two lens galaxies
#          can use the results of phases 1 and 2 to initialize their priors.
# Phase 4) Fit all relevant parameters simultaneously, using priors from phases 1, 2 and 3.

# Again, before we checkout the pipeline, lets import it, and get it running.
from howtolens.chapter_3_pipelines import tutorial_2_pipeline_x2_lens_galaxies

pipeline_x2_galaxies = tutorial_2_pipeline_x2_lens_galaxies.make_pipeline(
    phase_folders=["howtolens", "c3_t2_x2_galaxies"]
)

pipeline_x2_galaxies.run(dataset=imaging, mask=mask)

# Now, read through the '_tutorial_2_pipeline_x2_galaxies.py_' pipeline, to get a complete view of how it works.
# Once you've done that, come back here and we'll wrap up this tutorial.


# And, we're done. This pipeline takes a while to run, as is the nature of multi-galaxy modeling. Nevertheless, the
# techniques we've learnt above can be applied to systems with even more galaxies, albeit the increases in parameters
# will slow down the non-linear search. Here are some more Q&A's

# 1) This system had two very similar lens galaxies, with comparable amounts of light and mass. How common is this?
#    Does it make it harder to model them?

#    Typically, a 2 galaxy system has 1 massive galaxy (that makes up some 80%-90% of the overall light and mass),
#    accompanied by a smaller satellite. The satellite can't be ignored - it impacts the ray-tracing in a measureable
#    way, but its a lot less degenerate with the 'main' lens galaxy. This means we can often model the  satellite
#    with much simpler profiles (e.g. spherical profiles). So yes, multi-galaxy systems can often be easier to model.

# 2) It got pretty confusing passing all those priors towards the end of the pipeline there, didn't it?

#    It does get confusing, I won't lie. This is why we made galaxies named objects - so that we could call them the
#    'left_lens' and 'right_lens'. It still requires caution when writing the pipeline, but goes to show that if
#    you name your galaxies sensibly you should be able to avoid errors, or spot them quickly when you make them.
