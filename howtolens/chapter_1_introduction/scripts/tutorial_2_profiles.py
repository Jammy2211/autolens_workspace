from autolens.model.profiles import light_profiles
from autolens.model.profiles import mass_profiles
from autolens.array import grids
from autolens.model.profiles.plotters import profile_plotters

# In this example, we'll create a grid of Cartesian (y,x) coordinates and pass it to the 'light_profiles'
# module to create images on this grid and the 'mass_profiles' module to create deflection-angle maps on this grid.

# Lets use the same grid as the previous tutorial (if you skipped that tutorial, I recommend you go back to it!)
grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
    shape=(100, 100), pixel_scale=0.05, sub_grid_size=2
)

# Next, lets create a light profile using the 'light_profiles' module. We'll use a Sersic function,
# which is a analytic function often use to depict galaxies.
sersic_light_profile = light_profiles.EllipticalSersic(
    centre=(0.0, 0.0),
    axis_ratio=0.8,
    phi=45.0,
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

# We can print a profile to confirm its parameters.
print(sersic_light_profile)

# We can pass a grid to a light profile to compute its intensity at every grid coordinate. When we compute an
# array from a grid using a '_from_grid' method like the one below, we have two options for how the calculation is performed:

# 1) The values (e.g. the image) are calculated on the sub-grid. The function will either return the
#    values on this sub grid or bin the sub-gridded values to the grids actual shape (in this case (100, 100)).

# 2) The values are either mapped to the 2D shape of the grid (100, 100) or returned as a flattened 1D NumPy array,
#    on either the sub-grid (e.g. 40000 values) or non sub-gridded array (10000 values).

# This behaviour is determined by two boolean inputs, 'return_in_2d' and 'return_binned'.
light_profile_image = sersic_light_profile.profile_image_from_grid(
    grid=grid, return_in_2d=True, return_binned=True
)

# In this case, the array is returned in 2D and binned up to the shape of the input grid (100, 100).

print("intensity of central grid pixels:")
print(light_profile_image[49, 49])
print(light_profile_image[49, 50])
print(light_profile_image[50, 49])
print(light_profile_image[50, 50])

# We can return the binned-up array in 1D.
light_profile_image = sersic_light_profile.profile_image_from_grid(
    grid=grid, return_in_2d=False, return_binned=True
)

print("intensity of grid pixel 1:")
print(light_profile_image[0])
print("intensity of grid pixel 2:")
print(light_profile_image[1])
print()

# The 1D flattening occurs from the top-left pixel, and goes rightwards and downwards. Thus, because the
# light profile is centered at (0.0, 0.0), the central pixels are the brightest.
print("intensity of central grid pixels:")
print(light_profile_image[4949])
print(light_profile_image[4950])
print(light_profile_image[5049])
print(light_profile_image[5050])

# If we set return_binned to False, we return the values on the sub-grid, which for a sub_grid_size of 2 gives us
# 4 times as many values.

light_profile_image = sersic_light_profile.profile_image_from_grid(
    grid=grid, return_in_2d=False, return_binned=False
)
print("Number of sub pixels = ", light_profile_image.shape[0])

# By default, all arrays from a '_from_grid' method are returned in 2D and binned up, so if you don't specify these
# boolean inputs you'll get the result in the most intuitive format.
light_profile_image = sersic_light_profile.profile_image_from_grid(grid=grid)
print("intensity of central grid pixels:")
print(light_profile_image[49, 49])
print(light_profile_image[49, 50])
print(light_profile_image[50, 49])
print(light_profile_image[50, 50])

# The 'return_in_2d' and 'return_binned' inputs are on pretty much *every* _from_grid methods in PyAutoLens!

# We can use a profile plotter to plot this intensity map (this maps the grid to 2D before plotting).
profile_plotters.plot_image(light_profile=sersic_light_profile, grid=grid)

# To perform ray-tracing, we need to create a 'mass-profile'. A mass-profile is an analytic function that describes the
# distribution of mass in a galaxy, and therefore can be used to derive its surface-density, gravitational potential
# and most importantly, its deflection angles. For those unfamiliar with lensing, the deflection angles describe how
# light is bent by the mass-profile due to the curvature of space-time.

# Lets create a singular isothermal sphere (SIS) mass-profile using the 'mass-profiles' module.
sis_mass_profile = mass_profiles.SphericalIsothermal(
    centre=(0.0, 0.0), einstein_radius=1.6
)

print(sis_mass_profile)

# Just like above, we can pass a grid to a mass-profile to compute its deflection angles (still in 1D)

# (If you are new to gravitiational lensing, and are unclear on what a 'deflection-angle' means or what it is
# used for, then I'll explain all in tutorial 4 of this chapter. For now, just look at the pretty pictures
# they make, and worry about what they mean in tutorial 4!).

mass_profile_deflections = sis_mass_profile.deflections_from_grid(
    grid=grid, return_in_2d=True, return_binned=True
)

print("deflection-angles of grid pixel 0:")
print(mass_profile_deflections[0, 0])
print("deflection-angles of grid pixel 1:")
print(mass_profile_deflections[0, 1])
print()
print("deflection-angles of central grid pixels:")
print(mass_profile_deflections[49, 49])
print(mass_profile_deflections[49, 50])
print(mass_profile_deflections[50, 49])
print(mass_profile_deflections[50, 50])

# And again, a profile plotter can plot these deflection angles in 2D.
profile_plotters.plot_deflections_y(mass_profile=sis_mass_profile, grid=grid)

profile_plotters.plot_deflections_x(mass_profile=sis_mass_profile, grid=grid)

# Mass profiles have a range of other properties that are used for lensing calculations, a couple of which we've
# plotted images of below:

# Convergence - The surface mass density of the mass-profile in dimensionless units which are convenient for lensing calcuations.
# Potential - The gravitational of the mass-profile again in convenient dimensionless units.
# Magnification - Describes how much brighter each image-pixel appears due to focusing of light rays by the mass-profile.

# Extracting arrays of these quantities fom PyAutoLens is exactly the same as for the image and deflection angles above.
mass_profile_convergence = sis_mass_profile.convergence_from_grid(
    grid=grid, return_in_2d=True, return_binned=True
)

mass_profile_potential = sis_mass_profile.potential_from_grid(
    grid=grid, return_in_2d=True, return_binned=True
)

mass_profile_magnification = sis_mass_profile.magnification_from_grid(
    grid=grid, return_in_2d=True, return_binned=True
)

# Plotting them is equally straight forward.

profile_plotters.plot_convergence(mass_profile=sis_mass_profile, grid=grid)

profile_plotters.plot_potential(mass_profile=sis_mass_profile, grid=grid)

profile_plotters.plot_magnification(mass_profile=sis_mass_profile, grid=grid)

# Congratulations, you've completed your second PyAutoLens tutorial! Before moving on to the next one, experiment with
# PyAutoLens by doing the following:
#
# 1) Change the light profile's effective radius and Sersic index - how does the image's appearance change?
# 2) Change the mass profile's einstein radius - what happens to the deflection angles, potential and convergence?
# 3) Experiment with different light-profiles and mass-profiles in the light_profiles and mass_profiles modules.
#    In particular, use the EllipticalIsothermal profile to introduce ellipticity into a mass profile.
