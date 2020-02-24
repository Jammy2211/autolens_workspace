# Welcome to your first pipeline runner, which we'll use to run the tutorial 1 pipeline.
#
# In PyAutoLens, every pipeline is a standalone function which 'makes' a pipeline. We then pass data to this pipeline and
# run it. This keeps our pipelines and data separate, which is good practise, as it it encourages us to write pipelines
# that generalize to as many lenses as possible!

# So, lets begin by discussing the pipeline in this tutorial, which fits the lens and source galaxy of a strong lens.

# In chapter 2, we fitted a strong lens which included the contribution of light from the lens galaxy. We're going to
# fit this lens again (I promise, this is the last time!). However, now we're using pipelines, we can perform a
# completely different (and significantly faster) analysis.

# Load up the PDFs from the previous tutorial -
# 'howtolens/chapter_2_lens_modeling/output/t5_linking_phase_2/image/pdf_triangle.png'.

# This is a big triangle. As we fit models using more and more parameters, its only going to get bigger!

# As usual, you should notice some clear degeneracies between:

# 1) The size (effective_radius, R_l) and intensity (intensity, I_l) of the light profiles.
# 2) The mass normalization (einstein_radius, /Theta_m) and ellipticity (axis_ratio, q_m) of mass profiles.

# This isn't surprising. You can produce similar looking galaxies by trading out intensity for size, and you can
# produce similar mass distributions by compensating for a loss in lens mass by making it a bit less elliptical.

# What do you notice about the contours between the lens galaxy's light-profile and its mass-profile / the source
# galaxy's light profile? Look again.

# That's right - they're not degenerate. The covariance between these sets of parameters is minimal. Again, this makes
# sense - why would fitting the lens's light (which is an elliptical blob of light) be degenerate with fitting the
# source's light (which is a ring of light)? They look nothing like one another!

# So, as a newly trained lens modeler, what does the lack of covariance between these parameters make you think?
# Hopefully, you're thinking, why should I bother fitting the lens and source galaxy simultaneously? Surely we can
# find the right regions of non-linear parameter space by fitting each separately first? This is
# what we're going to do in this tutorial, using a pipeline composed of a modest 3 phases:

# Phase 1) Fit the lens galaxy's light, ignoring the source.

# Phase 2) Fit the source galaxy's light, ignoring the lens.

# Phase 3) Fit both simultaneously, using these results to initialize our starting location in parameter space.

### RUNNER FORMAT ###

# A runner begins by setting up PyAutoFit, in particular the paths to the autolens_workspace, config files and where output
# is stored. It is then followed by the setup of PyAutoLens, in particular where data is stored and loaded.

# From here on, we'll use the configs in the 'autolens_workspace/config' folder, which are the default configs used
# by all pipelines (e.g. not just this tutorial, but when you model your own images and lenses!).

# We'll also put the output in 'autolens_workspace/output', which is where output goes for an analysis.

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

# The same simulate function we used in chapter 2.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            phi=45.0,
            intensity=0.04,
            effective_radius=0.5,
            sersic_index=3.5,
        ),
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, einstein_radius=1.0
        ),
        shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalExponential(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=0.2
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


# Now lets Simulate the imaging datawhich we'll fit using the pipeline.
imaging = simulate()

# We need to choose our mask for the analysis. Given the lens light is present in the image we'll need to include
# all of its light in the central regions of the image, so lets use a circular mask.
mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)

# To make a pipeline, we call one long function which is written in its own Python module,
# '_tutorial_1_pipeline_lens__source.py_'. Before we check it out, lets get the pipeline running. To do this, we
# import the module and run its 'make_pipeline' function.

# When we run the make_pipeline function, we specify a'phase_folders' which structure the way our output is stored -
# for this pipeline this will output the dataset as:
# 'autolens_workspace/output/howtolens/c3_t1_lens_and_source/pipeline_name' (the pipeline name is specified in the
# pipeline).
from howtolens.chapter_3_pipelines import tutorial_1_pipeline_lens_and_source

pipeline_lens_and_source = tutorial_1_pipeline_lens_and_source.make_pipeline(
    phase_folders=["howtolens", "c3_t1_lens_and_source"]
)

# To run a pipeline, we simply use its 'run' function, passing it the dataset we want to run the pipeline on. Simple, huh?
pipeline_lens_and_source.run(dataset=imaging, mask=mask)

# Okay, good job, we're running our first pipeline in PyAutoLens! But what does it *actually* do? Well, to find that
# out, go to the script '_tutorial_1_pipeline_lens_and_source.py_', which contains a full description of the pipeline,
# as well as an overview of the tools we use to write the most general pipelines possible. Once you're done,
# come back to this pipeline runner script and we'll wrap up tutorial 1.


# And there we have it, a pipeline that breaks the analysis of the lens and source galaxy into 3 simple phases. This
# approach is much faster than fitting the lens and source simultaneously from the beginning. Instead of asking
# you questions at the end of this chapter's tutorials, I'm gonna give a Q&A - this'll hopefully get you thinking about
# how to approach pipeline writing.

# 1) Can this pipeline really be generalized to any lens? Surely the radii of the masks in phase 1 and 2 depend on the
#    lens and source galaxies?
#
#    Whilst this is true, we've chosen values of mask radii above that are 'excessive' and mask out a lot more of the
#    image than just the source (which, in terms of run-time, is desirable). Thus, provided you know the Einstein
#    radius distribution of your lens sample, you can choose masks radii that will masks out every source in your sample
#    adequately (and even if some of the source is still there, who cares? The fit to the lens galaxy will be okay).


# 2) What if my source galaxy isn't a ring of light? Surely my Annulus masks won't match it?
#
#    Just use the annulus anyway! Yeah, you'll masks out lots of image pixels with no source light, but remember, at
#    the beginning of the pipeline, *we don't care*. In phase 3, we'll use a large circular masks and do the fit
#    properly.
