import autofit as af
import autolens as al
import autolens.plot as aplt

# In the previous example, we used a non-linear search to infer the best-fit lens model of imaging-imaging of a strong lens.
# In this example, we'll get a deeper intuition of how a non-linear search works.

# First, I want to develop the idea of a 'parameter space'. Lets think of a function, like the simple function below:

# f(x) = x^2

# In this function, when we input a parameter x, it returns a value f(x). The mappings between different values of x
# and f(x) define a parameter space (and if you remember your high school math classes, you'll remember this parameter
# space is a parabola).

# A function can of course have multiple parameters:

# f(x, y, z) = x + y^2 - z^3

# This function has 3 parameters, x, y and z. The mappings between x, y and z and f(x, y, z) again define a parameter
# space, albeit now in 3 dimensions. Nevertheless, one could still picture this parameter space as some 3 dimensional
# curved surface.

# The process of computing a likelihood in PyAutoLens can be visualized in exactly the same way. We have a set of lens
# model parameters, which we input into PyAutoLens's 'likelihood function'. Now, this likelihood function isn't something
# that we can write down analytically and its inherently non-linear. But, nevertheless, it is a function;
# if we put the same set of lens model parameters into it, we'll compute the same likelihood.

# We can write our likelihood function as follows (using x_mp, y_mp, I_lp etc. as short-hand notation for the
# mass-profile and light-profile parameters):

# f(x_mp, y_mp, R_mp, x_lp, y_lp, I_lp, R_lp) = a likelihood from PyAutoLens's tracer and fit.

# The point is, like we did for the simple functions above, we again have a parameter space! It can't be written down
# analytically and its undoubtedly very complex and non-linear. Fortunately, we've already learnt how to search it, and
# find the solutions which maximize our likelihood function!

# Lets inspect the results of the last tutorial's non-linear search. We're going to look at what are called 'probably
# density functions' or PDF's for short. These represent where the highest likelihood regions of parameter space
# were found for each parameter.
#
# Navigate to the folder 'autolens_workspace/howtolens/chapter_2_lens_modeling/output_optimizer/t1_non_linear_search/image' and
# open the 'pdf_triangle.png' figure. The Gaussian shaped lines running down the diagonal of this triangle represent 1D
# estimates of the highest likelihood regions that were found in parameter space for each parameter.

# The remaining figures, which look like contour-maps, show the maximum likelihood regions in 2D between every parameter
# pair. We often see that two parameters are 'degenerate', whereby increasing one and decreasing the other leads to a
# similar likelihood value. The 2D PDF between the source galaxy's light-profile's intensity (I_l4) and effective
# radius (R_l4) shows such a degeneracy. This makes sense - making the source galaxy brighter and smaller is similar to
# making it fainter and bigger!

# So, how does PyAutoLens know where to look in parameter space? A parameter, say, the Einstein Radius, could in
# principle take any value between negative and positive infinity. AutoLens must of told it to only search regions of
# parameter space with 'reasonable' values (i.e. Einstein radii of around 1"-3").

# These are our 'priors' - which define where we tell the non-linear search to search parameter space. PyAutoLens uses
# two types of prior:

# 1) UniformPrior - The values of a parameter are randomly drawn between a lower and upper limit. For example, the
#                   orientation angle phi of a profile typically assumes a uniform prior between 0.0 and 180.0 degrees.

# 2) GaussianPrior - The values of a parameter are randomly drawn from a Gaussian distribution with a mean value
#                    and a width sigma. For example, an Einstein radius might assume a mean value of 1.0" and width
#                    of sigma = 1.0".
#

# The default priors on all parameters can be found by navigating to the 'config/priors/default' folder, and inspecting
# config files like light_profiles.ini. The convention is as follow:

# [EllipticalSersic]          # These are the priors used for an EllipticalSersic profile.
# effective_radius=u,0.0,2.0  # Its effective radius uses a UniformPrior with lower_limit=0.0, upper_limit=2.0
# sersic_index=g,4.0,2.0      # Its Sersic index uses a GaussianPrior with mean=4.0 and sigma=2.0

# Lets again setup the paths and config-overrides for this chapter, so the non-linear search runs fast.

# You need to change the path below to the chapter 1 directory.
chapter_path = "/path/to/user/autolens_workspace/howtolens/chapter_2_lens_modeling/"
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace/howtolens/chapter_2_lens_modeling/"

af.conf.instance = af.conf.Config(
    config_path=chapter_path + "configs/t2_parameter_space_and_priors",
    output_path=chapter_path + "output",
)

# This function simulates the dataset we'll fit in this tutorial - which is identical to the previous tutorial.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6),
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


# Again, lets create the simulated imaging dataset and mask.
imaging = simulate()

mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)

#  To change the priors on specific parameters, we create our galaxy models and then, simply, customize their priors.

lens = al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalIsothermal)
source = al.GalaxyModel(redshift=1.0, light=al.lp.SphericalExponential)

# To change priors, we use the 'prior' module of PyAutoFit (imported as af). These priors link our GalaxyModel to the
# non-linear search. Thus, it tells PyAutoLens where to search non-linear parameter space.

# These two lines change the centre of the lens galaxy's mass-profile to UniformPriors around the coordinates
# (-0.1", 0.1"). For real lens modeling, this might be done by visually inspecting the centre of emission of
# the lens galaxy's light.

# The word 'mass' corresponds to the word we used when setting up the GalaxyModel above.

lens.mass.centre_0 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

lens.mass.centre_1 = af.UniformPrior(lower_limit=-0.1, upper_limit=0.1)

# Lets also change the prior on the lens galaxy's einstein radius to a GaussianPrior centred on 1.4".
# For real lens modeling, this might be done by visually estimating the radius the lens's arcs / ring appear.

lens.mass.einstein_radius = af.GaussianPrior(mean=1.4, sigma=0.2)

# We can also customize the source galaxy - lets say we believe it is compact and limit its effective radius.

source.light.effective_radius = af.UniformPrior(lower_limit=0.0, upper_limit=0.3)

# We can now create this custom phase like we did a hyper phase before. If you look at the 'model.info'
# file in the output of the non-linear search, you'll see that the priors have indeed been changed.
custom_phase = al.PhaseImaging(
    phase_name="phase_t2_custom_priors",
    galaxies=dict(lens=lens, source=source),
    optimizer_class=af.MultiNest,
)

print(
    "MultiNest has begun running - checkout the autolens_workspace/howtolens/chapter_2_lens_modeling/output/2_custom_priors"
    "folder for live output of the results, images and lens model."
    "This Jupyter notebook cell with progress once MultiNest has completed - this could take some time!"
)

results_custom = custom_phase.run(dataset=imaging, mask=mask)

aplt.fit_imaging.subplot_fit_imaging(fit=results_custom.most_likely_fit)

print("MultiNest has finished run - you may now continue the notebook.")

# And, we're done. This tutorial had some pretty difficult concepts to wrap your head around. However, I can't
# emphasize enough how important it is that you develop an intuition for non-linear searches and the notion of a
# non-linear parameter space. Becoming good at lens modeling is all being able to navigate a complex, degenerate and
# highly non-linear parameter space! Luckily, we're going to keep thinking about this in the next set of tutorials,
# so if you're not feeling too confident yet, you will be soon!

# Before continuing to the next tutorial, I want you think about whether anything could go wrong when we search a
# non-linear parameter space. Is it possible that we won't find the highest likelihood lens model? Why might this be?
#
# Try and list 3 reasons why this might happen. In the next tutorial, we'll learn about just that - failure!
