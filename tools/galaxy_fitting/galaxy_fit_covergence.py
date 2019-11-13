import os
import numpy as np

# In this script, we're going to simulate a convergence profile using a set of mass-profiles and perform a direct
# fit to this convergence using a model galaxy. This uses a non-linear search and the fitting process is equivalent
# to fitting an image, but it bypasses the lens modeling process (e.g. the source reconstruction and use of actual
# imaging dataset).

# Whilst this may sound like an odd thing to do, there are reasons why one may wish to perform a direct fit to a
# derived light or mass profile quantity, like the convergence:

# 1) If the mass-profile(s) used to generate the galaxy that is fitted and the model galaxy are different, this fit
#    will inform us of how the mismatch between profiles leads to a different estimate of the inferred mass profile
#    properties.

# 2) By bypassing the lens modeling process, we can test what results we get whilst bypassing the potential systematics
#    that arise from a lens model fit (e.g due to the source reconstruction or quality of dataset_label).

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al

# First, we'll setup the grid we use to simulate a convergence profile.
grid = al.grid.uniform(shape_2d=(250, 250), pixel_scales=0.05, sub_size=2)

# Now lets create a galaxy, using a singular isothermal sphere.
galaxy = al.galaxy(
    redshift=0.5, mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
)

# Next, we'll generate its convergence profile. Note that, because we're using the galaxy convergence
# function the sub-grid of the grid that is passed to this function is used to over-sample the convergence.
# The convergence averages over this sub-grid, such that it is the shape of the image (250, 250).
convergence = galaxy.convergence_from_grid(grid=grid)

# Now, we'll set this convergence up as our 'galaxy_data', meaning that it is what we'll fit via a non-linear
# search phase. To perform a fit we need a noise-map to help define our chi-squared. Given we are fitting a direct
# lensing quantity the actual values of this noise-map arn't particularly important, so we'll just use a noise-map of
# all 0.1's
noise_map = al.array.full(
    fill_value=0.1, shape_2d=convergence.shape_2d, pixel_scales=convergence.pixel_scales
)
data = al.galaxy_data(
    image=convergence, noise_map=noise_map, pixel_scales=convergence.pixel_scales
)

# The fit will use a mask, which we setup like any other fit. Lets use a circular mask of 2.0"
def mask_function_circular(shape_2d, pixel_scales):
    return al.mask.circular(
        shape_2d=shape_2d, pixel_scales=pixel_scales, sub_size=1, radius_arcsec=2.0
    )


# We can now use a special phase, called a 'PhaseGalaxy', to fit this convergence with a model galaxy the
# mass-profiles of which we get to choose. We'll fit it with a singular isothermal sphere and should see we retrieve
# the input model above.
phase = al.PhaseGalaxy(
    phase_name="galaxy_convergence_fit",
    use_convergence=True,
    galaxies=dict(lens=al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)),
    mask_function=mask_function_circular,
    sub_size=4,
    optimizer_class=af.MultiNest,
)

phase.run(galaxy_data=[data])

# If you check your output folder, you should see that this fit has been performed and visualization specific to a
# convergence fit is output.

# Fits to an intensity map and gravitational potential are also possible. To do so, simply change the profile quantity
# that is simulated and edit the 'use_convergence' flag in the PhaseGalaxy above to the appropriate quantity.
# You can also fit deflection angle maps - however this requires a few small changes to this script, so we've create a
# 'galaxy_fit_deflections_al.Py' example script to show you how.
