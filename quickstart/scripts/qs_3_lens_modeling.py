# In lens modeling, we want to find the combination of light profiles and mass profiles which accurate reproduce the
# observed image. How do we go about doing this?

# To begin, we have to choose our lens model parametrization. We don't need to specify the values of its light and
# mass profiles (e.g. the centre, einstein_radius, etc.), only the profiles themselves. In this example, we'll use
# the following lens model:

# 1) An Elliptical Isothermal Profile (SIE) for the lens galaxy's mass.
# 2) An External Shear, which accounts for lens by line-of-sight galaxies near the lens.
# 3) An Elliptical Sersic light profile for the source galaxy's light.

# I'll let you into a secret - this is the same lens model I used to simulate the image we're going to fit.

# So, how do we infer the light and mass profile parameters that give a good fit to our dataset?

# We could randomly guess a lens model, corresponding to some random set of parameters. We could use this model to
# create a tracer and fit the image-simulator, and quantify how good the fit was using its likelihood. If we keep guessing
# lens models, eventually we'd find one that provides a good fit (i.e. a high likelihood) to the dataset!

# It may sound surprising, but this is the basis of how lens modeling works. However, we can do a lot better than random
# guesses. Instead, we track the likelihood of our previous guesses and guess, on average, more models which use
# combinations of light-profile and mass-profile parameters that gave higher likelihood solutions previously. If a set
# of parameters provided a good fit to the dataset, another set with similar values probably will too!

# This is called a 'non-linear search' and its a fairly tool used in science. In the howtolens lectures, we go into
# the details of how a non-linear search works. For the quick-start tutorial, we'll omit the nitty-gritty details.

# We're going to use a non-linear search called 'MultiNest'. I highly recommend it and find its great for lens
# modeling. However, lets not worry about the details of how MultiNest actually works. Instead, just
# picture that a non-linear search in PyAutoLens operates as follows:

# 1) Randomly guess a lens model and use its light-profiles and mass-profiles to set up a lens galaxy, source galaxy
#    and a tracer.

# 2) Use this tracer to perform a fit, which generates a model image and compares it to the observed strong lens image,
#    providing a likelihood.

# 3) Repeat this many times, using the likelihoods of previous fits (typically those with a high likelihood) to
#    find lens models with higher likelihoods.

# First, we load the imaging datawe're going to it, so you'll again need to change the path below to that of
# your autolens_workspace.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a directory path.
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

dataset_path = workspace_path + "dataset/imaging/imaging/lens_sie__source_sersic/"

imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    psf_path=dataset_path + "psf.fits",
    pixel_scales=0.1,
)

mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.imaging.subplot_imaging(imaging=imaging, mask=mask)

# To set up the non-linear search we use a GalaxyModel. This behaves like a Galaxy object. However, whereas for a
# Galaxy we manually specified the value of light and mass profile parameter, for a GalaxyModel these are inferred by
# the non-linear search.

# Lets model the lens galaxy with an SIE mass profile and External Shear (which is what it was simulated with).
lens_galaxy_model = al.GalaxyModel(
    redshift=0.5, mass=al.mp.EllipticalIsothermal, shear=al.mp.ExternalShear
)

# Lets model the source galaxy with an Elliptical Sersic light profile (again, what it was simulated with).
source_galaxy_model = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

# A phase takes our galaxy models and fits their parameters via a non-linear search (in this case, MultiNest).

phase = al.PhaseImaging(
    phase_name="phase_quick_start_example",
    galaxies=dict(lens_galaxy=lens_galaxy_model, source_galaxy=source_galaxy_model),
    optimizer_class=af.MultiNest,
)

# We manually change some of the MultiNest setup to speed up lens modeling. In the HowToLens lecture series we
# explain what changing these setup is *actually* doing.

phase.optimizer.const_efficiency_mode = True
phase.optimizer.n_live_points = 50
phase.optimizer.sampling_efficiency = 0.5

# To run the phase, we pass it the dataset we want to fit and the non-linear search begins! As the phase runs, a logger
# shows you the parameters of the best-fit model.

# Depending on the complexity of the model being fitted (e.g. the number of parameters) and resolution of the dataset
# a non-linear search can take a while to run. Maybe minutes, maybe hours, maybe days! The image below should only
# take 10 minutes or so. Whilst you wait, lets explore the autolens_workspace:

# 1) Checkout the 'autolens_workspace/output folder', where a folder 'phase_quick_start_example' containing the MultiNest
#    output is found, as well as images displaying the best-fit lens model. Visualization is output on-the-fly by
#    PyAutoLens, meaning you can watch the best-fit lens model improve as MultiNest runs in real time!

# 2) Checkout 'autolens_workspace/config'. These config files controls general PyAutoLens setup that control
#    visualization, default MultiNest setup and the priors of different light and mass profiles parmeters.

# 3) Look at 'autolens_workspace/dataset'. This is where the dataset we load and fit is stored (e.g. .fits files of the
#    image, noise-map and PSF).

# 4) Finally, checkout 'autolens_workspace/tools'. This contains Python scripts for standard lens tasks, such as
#    simuating strong lens images and drawing custom masks for lens fits.

print(
    "MultiNest has begun running. "
    "This Jupyter notebook cell will progress once MultiNest has completed - this could take some time!"
)

results = phase.run(dataset=imaging, mask=mask)

print("MultiNest has finished run - you may now continue the notebook.")

# The best-fit solution (i.e. the highest likelihood) is stored in the 'results', which we can plot as per usual.
aplt.fit_imaging.subplot_fit_imaging(fit=results.most_likely_fit)

# Congratulations, you've modeled your first lens with PyAutoLens!

# For the quick-start tutorial, that is the extent to which we describe how lens modeling works in PyAutoLens.
# From here, you can use template pipelines to fit your own strong lens dataset. The howtolens lecture series has
# 8 tutorials covering lens modeling in more detail - I heartily recommend you give it a look a some point in the future
# if you find yourself using PyAutoLens a lot!
