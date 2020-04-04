import os
import numpy as np

# In this script, we're going to simulate a convergence profile using a set of mass-profiles and perform a direct
# fit to this convergence using a model galaxy. This uses a non-linear search and the fitting process is equivalent
# to fitting an image, but it bypasses the lens modeling process (e.g. the source reconstruction and use of actual
# imaging data).

# Whilst this may sound like an odd thing to do, there are reasons why one may wish to perform a direct fit to a
# derived light or mass profile quantity, like the convergence:

# 1) If the mass-profile(s) used to generate the galaxy that is fitted and the model galaxy are different, this fit
#    will inform us of how the mismatch between profiles leads to a different estimate of the inferred mass profile
#    properties.

# 2) By bypassing the lens modeling process, we can test what results we get whilst bypassing potential systematics
#    that arise from a lens model fit (e.g due to the source reconstruction or quality of instrument).

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

### AUTOLENS + DATA SETUP ###

import autolens as al

# First, we'll setup the grid we use to simulate a convergence profile.
pixel_scale = 0.05
image_shape_2d = (250, 250)
grid = al.Grid.uniform(
    shape_2d=image_shape_2d, pixel_scale=pixel_scale, sub_grid_size=4
)

# Now lets create a galaxy, using a singular isothermal sphere.
galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
)

# Next, we generate the galaxy's convergence profile.
convergence = galaxy.convergence_from_grid(
    grid=grid,
)

# Now, we'll set this convergence up as our 'GalaxyData', which is the 'data' we fit via a non-linear search. To
# perform a fit we need a noise-map to define our chi-squareds. Given we are fitting a direct lensing quantity the
# actual values of this noise-map arn't particularly important, so we'll just use a noise-map of all 0.1's
noise_map = al.Array.full(
    fill_valu=0.1, shape_2d=image_shape_2d, pixel_scale=pixel_scale
)
data = al.GalaxyData(image=convergence, noise_map=noise_map, pixel_scale=pixel_scale)

# The fit will use a mask, which we setup like any other fit
mask = al.Mask.circular(shape=image_shape_2d, pixel_scale=pixel_scale, radius_arcsec=2.0
    )

# We can now use a 'PhaseGalaxy', to fit this convergence with a model galaxy, the mass-profiles of which we get to
# choose. We'll fit it with a singular isothermal sphere and thus should infer the input model above.
phase = al.PhaseGalaxy(
    phase_name="phase_galaxy_convergence_fit",
    use_convergence=True,
    galaxies=dict(
        lens=al.GalaxyModel(redshift=0.5, mass=al.mp.SphericalIsothermal)
    ),
    sub_grid_size=4,
    non_linear_class=af.MultiNest,
)

phase.run(galaxy_data=[data])

# If you check your output folder, you should see that this fit has been performed and visualization specific to a
# surface-density fit is output.

# Fits to an intensity map and gravitational potential are also possible. To do so, simply change the profile quantitiy
# that is simuulated and edit the 'use_convergence' flag in the GalaxyFitPhase above to the appropriate quantity.
# You can also fit deflection angle maps - however this requires a few small changes to this script, so we've create a
# 'galaxy_fit_deflections_al.Py' example script to show you how.