"""
Tutorial 3: Galaxies
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

We again use the same 2D grid as the previous tutorials.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Galaxies__

A `Galaxy` is a collection of light and / or mass profiles at the same redshift.

Lets make a galaxy with a Sersic light, by create the light profile and passing it to a `Galaxy` object.
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
The galaxy has an `image_2d_from` method, which uses its light profiles to create an image of the galaxy.

This behaves identically to the `image_2d_from` method we saw for light profiles in the previous tutorial.
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
It also has a method for computing its image in 1D, like we saw for light profiles in the previous tutorial.
"""
galaxy_image_1d = galaxy_with_light_profile.image_1d_from(grid=grid)
print(galaxy_image_1d)

"""
A `GalaxyPlotter` allows us to the plot the image in 1D and 2D. 

Once again, the API is identical to the `LightProfilePlotter` we saw in the previous tutorial.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_light_profile, grid=grid)
galaxy_plotter.figures_2d(image=True)
galaxy_plotter.figures_1d(image=True)

"""
__Multiple Profiles__

The galaxy above had a single light profile, and therefore the calculation of its image was no different calculations
using just the light profile on its own.

However, a `Galaxy` can be composed of multiple light profiles.

Lets create a galaxy with three light profiles.
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
By plotting the galaxy's image we see a superposition of 3 blobs of light.

The image of multiple light profiles is simply the sum of their individual images. Therefore, all the galaxy
is really doing is summing the images of its light profiles.
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_3_light_profiles, grid=grid)
galaxy_plotter.figures_2d(image=True)

"""
The previous tutorial discussed how the light distributions of galaxies are closer to a log10 distribution than a 
linear one and showed a convenience method to plot the image in log10 space.

When plotting multiple galaxies, plotting in log10 space makes it easier to see by how much the galaxy images
overlap and blend with one another. 
"""
galaxy_plotter = aplt.GalaxyPlotter(
    galaxy=galaxy_with_3_light_profiles,
    grid=grid,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
galaxy_plotter.figures_2d(image=True)

"""
We can plot each individual light profile image using the `subplot_of_light_profiles` method.
"""
galaxy_plotter.subplot_of_light_profiles(image=True)

"""
We can plot all light profiles in 1D on the same figure.

This allows us to compare radially how the intensity of each light profile changes with radius, and therefore
on what scales each light profile emits the majority of light.

1D plots use grids aligned with each individual light profile centre, therefore 1D plots do visually show how these 
3 galaxies are misaligned in 2D.
"""
galaxy_plotter.figures_1d_decomposed(image=True)

"""
Mass profiles can be passed to a `Galaxy` object in the exact same way as light profiles. 

Lets create a galaxy with three isothermal mass profiles.
"""
mass_profile_1 = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)

mass_profile_2 = al.mp.IsothermalSph(centre=(1.0, 1.0), einstein_radius=1.0)

mass_profile_3 = al.mp.IsothermalSph(centre=(1.0, -1.0), einstein_radius=1.0)

galaxy_with_3_mass_profile_list = al.Galaxy(
    redshift=0.5, mass_1=mass_profile_1, mass_2=mass_profile_2, mass_3=mass_profile_3
)

print(galaxy_with_3_mass_profile_list)

"""
We can use a `GalaxyPlotter` to plot the deflection angles of this galaxy and its superposition of three isothermal 
mass profiles. 

We saw that the images of light profiles are simply the sum of each individual light profile. The same is true of
deflection angles of mass profiles.

The "critical curves", given by the yellow and / or white lines, go pretty crazy when we start adding more and more
mass profiles to a galaxy!
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_3_mass_profile_list, grid=grid)
galaxy_plotter.figures_2d(deflections_y=True, deflections_x=True)

"""
The convergence and potential are also the sum of each individual mass profile and are easy to plot with a galaxy.
"""
galaxy_plotter.figures_2d(convergence=True, potential=True)
galaxy_plotter.figures_1d(convergence=True, potential=True)

"""
The mass distributions of galaxies are also easier to see separated in log10 space.
"""
galaxy_plotter = aplt.GalaxyPlotter(
    galaxy=galaxy_with_3_mass_profile_list,
    grid=grid,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
galaxy_plotter.figures_2d(convergence=True, potential=True)

"""
__Light + Mass Profiles__

A `Galaxy` can take both light and mass profiles, and there is no limit to how many we pass it!
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
The galaxy has a total of 4 light profiles and 4 mass profiles. 

Their image, convergence, potential and deflections angles all look pretty interesting!
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=galaxy_with_many_profiles, grid=grid)
galaxy_plotter.figures_2d(
    image=True, convergence=True, potential=True, deflections_y=True, deflections_x=True
)

"""
__Multiple Galaxies__

We can also group galaxies into a `Galaxies` object, which is constructed from a list of galaxies.
"""
galaxies = al.Galaxies(
    galaxies=[galaxy_with_light_profile, galaxy_with_3_light_profiles]
)

"""
The galaxies has the same methods we've seen for light profiles, mass profiles and individual galaxies.

For example, the `image_2d_from` method sums up the individual images of every galaxy.
"""
image = galaxies.image_2d_from(grid=grid)

"""
The `GalaxiesPlotter` shares the same API as the `LightProfilePlotter`, `MassProfilePlotter` and `GalaxyPlotter`.
"""
galaxies_plotter = aplt.GalaxiesPlotter(galaxies=galaxies, grid=grid)
galaxies_plotter.figures_2d(image=True)

"""
A subplot can be made of each individual galaxy image.
"""
galaxies_plotter.subplot_galaxy_images()

"""
In the next tutorial, we will see how grouping galaxies together is key to performing strong lensing calculations.

Notationally a group of galaxies is referred to as a plane.

__Wrap Up__

Tutorial 3 complete! Lets finish with just one question:

 1) We've learnt that by grouping light and mass profiles into a galaxy we can sum the contribution of each profile to 
 compute the galaxy's image, convergence, deflection angles, etc. 
 
 In strong lensing, there may be multiple galaxies (at the same redshift) next to one another. How might we combine 
 these galaxies to calculate their light and mass profile quantities?
"""
