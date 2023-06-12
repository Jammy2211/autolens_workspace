"""
Tutorial 3: galaxies
====================

This tutorial introduces `Galaxy` objects, which:

 - Are composed from collections of the light and mass profiles introduced in the previous tutorial.
 - Combine these profiles such that their properties (e.g. an image, deflection angles) are correctly calculated
 as the combination of these profiles.
 - Also have a redshift, which defines where a galaxy is relative to other galaxies in a lensing calculation.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

Lets use the same `Grid2D` as the previous tutorial.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Galaxies__

Lets make a galaxy with an elliptical Sersic `LightProfile`, by simply passing this profile to a `Galaxy` object.
"""
sersic_light_profile = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

galaxy_with_light_profile = al.Galaxy(redshift=0.5, light=sersic_light_profile)

print(galaxy_with_light_profile)

"""
We have seen that we can pass a 2D grid to a light profile to compute its 2D image via its `image_2d_from` method. We 
can do the exact same with a galaxy:
"""
galaxy_image_2d = galaxy_with_light_profile.image_2d_from(grid=grid)

print("intensity of `Grid2D` pixel 0:")
print(galaxy_image_2d.native[0, 0])
print("intensity of `Grid2D` pixel 1:")
print(galaxy_image_2d.native[0, 1])
print("intensity of `Grid2D` pixel 2:")
print(galaxy_image_2d.native[0, 2])
print("etc.")

"""
We can do the same for its 1D image:
"""
galaxy_image_1d = galaxy_with_light_profile.image_1d_from(grid=grid)
print(galaxy_image_1d)

"""
A `GalaxyPlotter` allows us to the plot the image in 1D and 2D, just like the `LightProfilePlotter` did for a light 
profile.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_light_profile, grid=grid)
galaxy_plotter.figures_2d(image=True)
galaxy_plotter.figures_1d(image=True)

"""
__Multiple Profiles__

We can pass galaxies as many light profiles as we like to a `Galaxy`, so lets create a galaxy with three light profiles.
"""
light_profile_1 = al.lp.SersicSph(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=2.5
)

light_profile_2 = al.lp.SersicSph(
    centre=(1.0, 1.0), intensity=1.0, effective_radius=2.0, sersic_index=3.0
)

light_profile_3 = al.lp.SersicSph(
    centre=(1.0, -1.0), intensity=1.0, effective_radius=2.0, sersic_index=2.0
)

galaxy_with_3_light_profiles = al.Galaxy(
    redshift=0.5,
    light_1=light_profile_1,
    light_2=light_profile_2,
    light_3=light_profile_3,
)

print(galaxy_with_3_light_profiles)

"""
If we plot the galaxy, we see 3 blobs of light!

(The image of multiple light profiles is simply the sum of the image of each individual light profile).
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_3_light_profiles, grid=grid)
galaxy_plotter.figures_2d(image=True)

"""
We can also plot each individual `LightProfile` using the plotter's `subplot_of_light_profiles` method.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
We can plot all light profiles in 1D, showing their decomposition of how they make up the overall galaxy.

Remember that 1D plots use grids aligned with each individual light profile centre, thus the 1D plot does not
show how these 3 galaxies are misaligned in 2D.
"""
galaxy_plotter.figures_1d_decomposed(image=True)

"""
We can pass mass profiles to a `Galaxy` object in the exact same way as light profiles. Lets create a `Galaxy` with 
three spherical isothermal mass profile's. 
"""
mass_profile_1 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)

mass_profile_2 = al.mp.IsothermalSph(centre=(1.0, 1.0), einstein_radius=1.0)

mass_profile_3 = al.mp.IsothermalSph(centre=(1.0, -1.0), einstein_radius=1.0)

galaxy_with_3_mass_profile_list = al.Galaxy(
    redshift=0.5, mass_1=mass_profile_1, mass_2=mass_profile_2, mass_3=mass_profile_3
)

print(galaxy_with_3_mass_profile_list)

"""
We can use a `GalaxyPlotter` to plot the deflection angles of this galaxy, which is the deflection angles due to 
three separate spherical isothermal mass profiles. 

(The deflection angles of multiple mass profiles are simply the sum of the deflection angles of each individual mass
profile).
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_3_mass_profile_list, grid=grid)
galaxy_plotter.figures_2d(deflections_y=True, deflections_x=True)

"""
I wonder what 3 summed convergence maps or potential`s look like ;).

(These are again the sum of the individual mass profile convergences or potentials).
"""
galaxy_plotter.figures_2d(convergence=True, potential=True)
galaxy_plotter.figures_1d(convergence=True, potential=True)

"""
Finally, a `Galaxy` can take both light and mass profiles, and there is no limit to how many we pass it!
"""
light_profile_1 = al.lp.SersicSph(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
)

light_profile_2 = al.lp.SersicSph(
    centre=(1.0, 1.0), intensity=1.0, effective_radius=2.0, sersic_index=2.0
)

light_profile_3 = al.lp.SersicSph(
    centre=(2.0, 2.0), intensity=1.0, effective_radius=3.0, sersic_index=3.0
)

light_profile_4 = al.lp.Sersic(
    centre=(1.0, -1.0),
    ell_comps=(0.3, 0.0),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=1.0,
)

mass_profile_1 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)

mass_profile_2 = al.mp.IsothermalSph(centre=(1.0, 1.0), einstein_radius=2.0)

mass_profile_3 = al.mp.IsothermalSph(centre=(2.0, 2.0), einstein_radius=3.0)

mass_profile_4 = al.mp.Isothermal(
    centre=(1.0, -1.0), ell_comps=(0.333333, 0.0), einstein_radius=2.0
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

"""
Suffice to say, this `Galaxy`'s images, convergence, potential and deflections look pretty interesting.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_many_profiles, grid=grid)
galaxy_plotter.figures_2d(
    image=True, convergence=True, potential=True, deflections_y=True, deflections_x=True
)

"""
__Wrap Up__

Tutorial 3 complete! Lets finish with just one question:

 1) We've learnt that by grouping light and mass profiles into a galaxy we can sum the contribution of each profile to 
 compute the galaxy's image, convergence, deflection angles, etc. 
 
 In strong lensing, there may be multiple galaxies (at the same redshift) next to one another. How might we combine 
 these galaxies to calculate their light and mass profile quantities?
"""
