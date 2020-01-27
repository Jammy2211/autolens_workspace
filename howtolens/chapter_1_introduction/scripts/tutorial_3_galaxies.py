import autolens as al
import autolens.plot as aplt

# In this example, we'll use the 'mass_profiles' and 'light_profiles' modules introduced previously, along with the
# 'galaxy' module to create Galaxy objects in PyAutoLens. We'll see that:

# 1) Galaxies can be made from multiple light-profiles and mass-profiles.
# 2) By taking multiple components, the summed image / deflection angle's of the profiles are computed.
# 3) Galaxies have redshifts, defining where they are relative to one another in lens calculations.

# Lets use an identical grid to the previous example.
grid = al.grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

# Lets make a galaxy with a Sersic light profile, by making a Sersic light profile and pasing it to a Galaxy object.
sersic_light_profile = al.lp.EllipticalSersic(
    centre=(0.0, 0.0),
    axis_ratio=0.8,
    phi=45.0,
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

galaxy_with_light_profile = al.Galaxy(redshift=0.5, light=sersic_light_profile)

# We can print the galaxy to confirm its profile and its parameters
print(galaxy_with_light_profile)

# In the previous example, we passed grids to the light-profile module to compute its image.
# We can do the exact same with galaxies, to again compute the galaxy's image.
galaxy_image = galaxy_with_light_profile.profile_image_from_grid(grid=grid)

print("intensity of grid pixel 0:")
print(galaxy_image[0, 0])
print("intensity of grid pixel 1:")
print(galaxy_image[0, 1])
print("intensity of grid pixel 2:")
print(galaxy_image[0, 2])
print("etc.")

# A galaxy plotter allows us to the plotters the image, just like the profile plotters did for a light profile.
aplt.galaxy.profile_image(galaxy=galaxy_with_light_profile, grid=grid)

# We can pass galaxies as many profiles as we like. Lets create a galaxy with three light profiles.
light_profile_1 = al.lp.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=2.5
)

light_profile_2 = al.lp.SphericalSersic(
    centre=(1.0, 1.0), intensity=1.0, effective_radius=2.0, sersic_index=3.0
)

light_profile_3 = al.lp.SphericalSersic(
    centre=(1.0, -1.0), intensity=1.0, effective_radius=2.0, sersic_index=2.0
)

galaxy_with_3_light_profiles = al.Galaxy(
    redshift=0.5,
    light_1=light_profile_1,
    light_2=light_profile_2,
    light_3=light_profile_3,
)

# We can print the galaxy to confirm it possesses the Sersic light-profiles above.
print(galaxy_with_3_light_profiles)

# If we plotters the galaxy, we see 3 blobs of light!
aplt.galaxy.profile_image(galaxy=galaxy_with_3_light_profiles, grid=grid)

# We can also plotters each individual light profile using the 'subplot' galaxy plotter.
aplt.galaxy.profile_image_subplot(galaxy=galaxy_with_3_light_profiles, grid=grid)

# Mass profiles interact with Galaxy objects in the exact same way as light profiles.
# Lets create a galaxy with three SIS mass profiles.
mass_profile_1 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

mass_profile_2 = al.mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=1.0)

mass_profile_3 = al.mp.SphericalIsothermal(centre=(1.0, -1.0), einstein_radius=1.0)

galaxy_with_3_mass_profiles = al.Galaxy(
    redshift=0.5, mass_1=mass_profile_1, mass_2=mass_profile_2, mass_3=mass_profile_3
)

# We can print a galaxy to confirm it possesses the sis mass-profiles above.
print(galaxy_with_3_mass_profiles)

# We can use a galaxy plotter to plot these deflection angles.

# (Deflection angles of mass-profiles add together just like the light-profile image's above)
aplt.galaxy.deflections_y(galaxy=galaxy_with_3_mass_profiles, grid=grid)
aplt.galaxy.deflections_x(galaxy=galaxy_with_3_mass_profiles, grid=grid)

# I wonder what 3 summed convergence profiles or potential's look like ;)
aplt.galaxy.convergence(galaxy=galaxy_with_3_mass_profiles, grid=grid)
aplt.galaxy.potential(galaxy=galaxy_with_3_mass_profiles, grid=grid)

# Finally, a galaxy can take both light and mass profiles, and there is no limit to how many we pass it.
light_profile_1 = al.lp.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
)

light_profile_2 = al.lp.SphericalSersic(
    centre=(1.0, 1.0), intensity=1.0, effective_radius=2.0, sersic_index=2.0
)

light_profile_3 = al.lp.SphericalSersic(
    centre=(2.0, 2.0), intensity=1.0, effective_radius=3.0, sersic_index=3.0
)

light_profile_4 = al.lp.EllipticalSersic(
    centre=(1.0, -1.0),
    axis_ratio=0.5,
    phi=45.0,
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=1.0,
)

mass_profile_1 = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

mass_profile_2 = al.mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=2.0)

mass_profile_3 = al.mp.SphericalIsothermal(centre=(2.0, 2.0), einstein_radius=3.0)

mass_profile_4 = al.mp.EllipticalIsothermal(
    centre=(1.0, -1.0), axis_ratio=0.5, phi=45.0, einstein_radius=2.0
)

galaxy_with_many_profiles = al.Galaxy(
    redshift=0.5,
    light_1=light_profile_1,
    light_2=light_profile_2,
    light_3=light_profile_3,
    light_4=light_profile_4,
    mass_1=mass_profile_1,
    mass_2=mass_profile_2,
    mass_3=mass_profile_3,
    mass_4=mass_profile_4,
)

# Suffice to say, the galaxy's images, convergence, potential and deflections look pretty
# interesting.

aplt.galaxy.profile_image(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.galaxy.convergence(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.galaxy.potential(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.galaxy.deflections_y(galaxy=galaxy_with_many_profiles, grid=grid)

aplt.galaxy.deflections_x(galaxy=galaxy_with_many_profiles, grid=grid)

# And we're done. Lets finished by just thinking about one question:

# 1) We've learnt we can group profiles into galaxies, to essentially sum the contribution of each light profile to the
#    galaxy image's intensity, or sum the contribution of each mass profile to the convergence, potential and
#    deflection angles. In strong lens, there are often multiple galaxies next to one another responsible for the
#    lens - how might we account for this?
