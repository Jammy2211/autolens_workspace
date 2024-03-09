"""
Tutorial 2: Profiles
====================

This tutorial introduces light profile and mass objects, wherre:

 - `LightProfile` represents analytic forms for the light distribution of galaxies.
 - `MassProfile`: represents analytic forms for the mass distributions of galaxies.

By passing these objects 2D grids of $(y,x)$ coordinates we can create images from a light profile and deflection
angle maps from a mass profile, the latter of which will ultimately describe how light is ray-traced throughout the
Universe by a strong gravitational lens!
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

We setup a 2D grid with the same resolution and arc-second to pixel conversion as the previous tutorial.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Light Profiles__

We now create a light profile using the `light_profile` module, which is accessible via `al.lp`.

We'll use the elliptical Sersic light profile, using the `Sersic` object, which is an analytic function used 
throughout studies of galaxy morphology to represent their light. 

This profile is elliptical and we'll use the `ell_comps` to describe its elliptical geometry. If you are unsure what 
the `ell_comps` are, I'll give a description of them at the end of the tutorial.
"""
sersic_light_profile = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

"""
By printing a `LightProfile` we can display its parameters.
"""
print(sersic_light_profile)

"""
__Images__

We next pass the grid to the `sersic_light_profile`, to compute the intensity of the Sersic at every (y,x) 
coordinate on our two dimension grid. 

This uses the `image_2d_from` method, one of many `_from` methods that **PyAutoLens** uses to compute quantities from 
a grid.
"""
image = sersic_light_profile.image_2d_from(grid=grid)

"""
Similar to the `Grid2D` objects discussed in the previous tutorial, this returns an `Array2D` object:
"""
print(type(image))

"""
Like the grid, the `Array2D` object has both `native` and `slim` attributes:
"""
print("Intensity of pixel 0:")
print(image.native[0, 0])
print("Intensity of pixel 1:")
print(image.slim[1])

"""
For an `Array2D`, the dimensions of these attributes are as follows:

 - `native`: an ndarray of shape [total_y_image_pixels, total_x_image_pixels].

 - `slim`: an ndarray of shape [total_y_image_pixels*total_x_image_pixels].

The `native` and `slim` dimensions are therefore analogous to those of the `Grid2D` object, but without the final 
dimension of 2.
"""
print(image.shape_native)
print(image.shape_slim)

"""
We can use a `LightProfilePlotter` to plot the image of a light profile. 

We pass this plotter the light profile and a grid, which are used to create the image that is plotted.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile, grid=grid
)
light_profile_plotter.figures_2d(image=True)

"""
The light distributions of galaxies are closer to a log10 distribution than a linear one. 

This means that when we plot an image of a light profile, its appearance is better highlighted when we take the
logarithm of its values and plot it in log10 space.

The `MatPlot2D` object has an input `use_log10`, which will do this automatically when we call the `figures_2d` method.

Below, we can see that the image plotted now appears more clearly, with the outskirts of the light profile more visible.
"""
light_profile_plotter = aplt.LightProfilePlotter(
    light_profile=sersic_light_profile,
    grid=grid,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
light_profile_plotter.figures_2d(image=True)

"""
We can also compute and plot 1D quantities of the light profile, which show how the image intensity varies radially.

1D plots use a radial grid which is aligned with the profile centre and major-axis.
"""
print(sersic_light_profile.image_1d_from(grid=grid))

light_profile_plotter.figures_1d(image=True)

"""
__Mass Profiles__

To perform lensing calculations we use mass profiles using the `mass_profile` module, which is accessible via `al.mp`.

A mass profile is an analytic function that describes the distribution of mass in a galaxy. It can therefore be used 
to derive its surface-density, gravitational potential and, most importantly, its deflection angles. 

In gravitational lensing, the deflection angles describe how mass deflections light due to how it curves space-time.

We use `Sph` to concisely describe that this profile is spherical.
"""
sis_mass_profile = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)

print(sis_mass_profile)

"""
__Deflection Angles__

We can again use a `from_grid_` method to compute the deflection angles of a mass profile from a grid. 

The deflection angles are returned as the arc-second deflections of the grid's $(y,x)$ Cartesian components. As seen
for grids and arrays, we can access the deflection angles via the `native` and `slim` attributes. 

In anything is unclear, in tutorial 4 it will become clear how these deflection angles are used to perform strong 
gravitational lensing calculations.
"""
mass_profile_deflections_yx_2d = sis_mass_profile.deflections_yx_2d_from(grid=grid)

print("deflection-angles of `Grid2D` pixel 0:")
print(mass_profile_deflections_yx_2d.native[0, 0])
print("deflection-angles of `Grid2D` pixel 1:")
print(mass_profile_deflections_yx_2d.slim[1])
print()

"""
A `MassProfilePlotter` can plot the deflection angles, which are plotted separately for the y and x components.

Overlaid on this figure and many other mass profile figures are yellow and white lines, which are called 
the "critical curves".  

These are an important concept in lensing, and we will explain what they are in tutorial 5.
"""
mass_profile_plottter = aplt.MassProfilePlotter(
    mass_profile=sis_mass_profile, grid=grid
)
mass_profile_plottter.figures_2d(deflections_y=True, deflections_x=True)

"""
__Other Properties__

Mass profiles have a range of other properties that are used for lensing calculations:

 - `convergence`: The surface mass density of the mass profile in dimensionless units.
 - `potential`: The "lensing potential" of the mass profile in dimensionless units.
 - `magnification`: How much brighter light ray appear due to magnification and the focusing of light rays.

These can all be calculated using the `*_from` methods and are returned as `Array2D`'s.
"""
convergence_2d = sis_mass_profile.convergence_2d_from(grid=grid)
potential_2d = sis_mass_profile.potential_2d_from(grid=grid)
magnification_2d = sis_mass_profile.magnification_2d_from(grid=grid)

"""
One dimensional versions of these quantities can also be computed showing how they vary radially from the centre of the
profile.
"""
convergence_1d = sis_mass_profile.convergence_1d_from(grid=grid)
potential_1d = sis_mass_profile.potential_1d_from(grid=grid)

"""
The same plotter API used previous can be used to plot these quantities.
"""
mass_profile_plottter.figures_2d(convergence=True, potential=True, magnification=True)
mass_profile_plottter.figures_1d(convergence=True, potential=True)

"""
The convergence and potential are also quantities that are better plotted in log10 space.
"""
mass_profile_plottter = aplt.MassProfilePlotter(
    mass_profile=sis_mass_profile, grid=grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
mass_profile_plottter.figures_2d(convergence=True, potential=True)

"""
This tutorial has introduced a number of lensing quantities and you may be unsure what they and what their use is,
for example the critical curves, convergence, potential and magnification.

These will be described in detail at the end of chapter 1 of the **HowToLens** lectures. 

Before we get there, the tutorials will first focus on using just the deflection angles of mass profiles to illustrate 
how gravitational lensing ray-tracing works.

__Wrap Up__

Congratulations, you`ve completed your second **PyAutoLens** tutorial! 

Before moving on to the next one, experiment by doing the following:

1) Change the `LightProfile`'s effective radius and Sersic index - how does the image's appearance change?
2) Change the `MassProfile`'s einstein radius - what happens to the deflection angles, potential and convergence?
3) Experiment with different `LightProfile`'s and `MassProfile`'s in the `light_profile` and `mass_profile` modules. 
In particular, try the `Isothermal` `Profile`, which introduces ellipticity into the mass distribution

___Elliptical Components___

The `ell_comps` describe the ellipticity of light and mass distributions. 

We can define a coordinate system where an ellipse is defined in terms of:

 - axis_ratio = semi-major axis / semi-minor axis = b/a
 - position angle, where angle is in degrees.

See https://en.wikipedia.org/wiki/Ellipse for a full description of elliptical coordinates.

The elliptical components are related to the axis-ratio and position angle as follows:

    fac = (1 - axis_ratio) / (1 + axis_ratio)
    
    elliptical_comp[0] = elliptical_comp_y = fac * np.sin(2 * angle)
    elliptical_comp[1] = elliptical_comp_x = fac * np.cos(2 * angle)

We can use the **PyAutoLens** `convert` module to determine the elliptical components from an `axis_ratio` and `angle`,
noting that the position angle is defined counter-clockwise from the positive x-axis.
"""
ell_comps = al.convert.ell_comps_from(axis_ratio=0.5, angle=45.0)

print(ell_comps)

"""
The reason light profiles and mass profiles use the elliptical components instead of an axis-ratio and position angle is
because it improves the lens modeling process. What is lens modeling? You'll find out in chapter 2!
"""
