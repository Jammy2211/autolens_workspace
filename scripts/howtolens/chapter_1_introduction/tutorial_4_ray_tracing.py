"""
Tutorial 4: Ray Tracing
=======================

In this tutorial, we use combinations of light profiles, mass profiles and galaxies to perform our first ray-tracing 
calculations!

A strong gravitational lens is a system where two (or more) galaxies align perfectly down our line of sight from Earth
such that the foreground galaxy's mass (represented as mass profiles) deflects the light (represented as light profiles)
of a background source galaxy(s).

When the alignment is just right and the lens is massive enough, the background source galaxy appears multiple
times. The schematic below shows such a system, where light-rays from the source are deflected around the lens galaxy
to the observer following multiple distinct paths.

![Schematic of Gravitational Lensing](https://i.imgur.com/zB6tIdI.jpg)

As an observer, we don't see the source's true appearance (e.g. a round blob of light). Instead, we only observe its
light after it has been deflected and lensed by the foreground galaxies.

In the schematic above, we used the terms 'image-plane' and 'source-plane'. In lensing, a 'plane' is a collection of
galaxies at the same redshift (meaning that they are physically parallel to one another). In this tutorial, we'll
create a strong lensing system made-up of planes, like the one pictured above. Whilst a plane can contain
any number of galaxies, in this tutorial we'll stick to just one lens galaxy and one source galaxy.

In this tutorial, we therefore introduce potentially the most important object, the `Tracer`. This exploits the
redshift information of galaxies to automatically perform ray-tracing calculations.
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

We should now think of this grid as the coordinates we will "trace" from the image-plane to the source-plane.

We therefore name it the `image_plane_grid`.
"""
image_plane_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
We next create galaxies, made up of light and mass profiles, which we will use to perform ray-tracing.

We name them `lens_galaxy` and `source_galaxy`, to reflect their role in the lensing schematic above.

The redshifts of the galaxies now take on more significance, as they are used when we perform ray-tracing calculations
below to determine the order of calculations.
"""
sis_mass_profile = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)

lens_galaxy = al.Galaxy(redshift=0.5, mass=sis_mass_profile)

print(lens_galaxy)

sersic_light_profile = al.lp.SersiclCoreSph(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
)

source_galaxy = al.Galaxy(redshift=1.0, light=sersic_light_profile)

print(source_galaxy)

"""
__Tracer__

We now use use the lens and source galaxies to perform lensing and ray-tracing calculations.

When we pass our galaxies into the `Tracer` below, the following happens:

1) The galaxies are ordered in ascending redshift.
2) The galaxies are grouped in a list at every unique redshift in a "plane".
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
The tracer has a `planes` attributes, where each plane is a group of all galaxies at the same redshift.

This simple lens system has just two galaxies at two unique redshifts, so the tracer has two planes. 

The planes list therefore has length 2, with the first entry being the image-plane and the second entry being the
source-plane.
"""
print(tracer.planes)
print(len(tracer.planes))

"""
We can print each plane, which shows the galaxies that it contains.
 
The contents of each plane is the `Galaxies` object we introduced in the previous tutorial. 
"""
print("Image Plane:")
print(tracer.planes[0])
print()
print("Source Plane:")
print(tracer.planes[1])

"""
This allows us to perform calculations for each plane individually. 

For example we can calculate and plot the deflection angles of all image-plane galaxies 
"""
deflections_yx_2d = tracer.planes[0].deflections_yx_2d_from(grid=image_plane_grid)

print("deflection-angles of `Plane`'s `Grid2D` pixel 0:")
print(deflections_yx_2d.native[0, 0, 0])
print(deflections_yx_2d.native[0, 0, 0])

print("deflection-angles of `Plane`'s `Grid2D` pixel 1:")
print(deflections_yx_2d.native[0, 1, 1])
print(deflections_yx_2d.native[0, 1, 1])

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=tracer.planes[0], grid=image_plane_grid
)
galaxies_plotter.figures_2d(deflections_y=True, deflections_x=True)

"""
__Manual Ray Tracing__

We have frequently plotted the deflection angles of mass profiles in this chapter, but we are yet to actually use
them to perform ray-tracing!

The deflection angles tell us how light is "deflected" by the lens galaxy. 

By subtracting the $(y,x)$ grid of deflection angles from the $(y,x)$ grid of image-plane coordinates we can determine
how the mass profile deflections light and in turn compute the source plane coordinates:

 `source_plane_coordinates = image_plane_coordinates - image_plane_deflection_angles`

We perform this below using the `traced_grid_2d_from` method of the image-plane galaxies:
"""
source_plane_grid = tracer.planes[0].traced_grid_2d_from(grid=image_plane_grid)

print("Traced source-plane coordinates of `Grid2D` pixel 0:")
print(source_plane_grid.native[0, 0, :])
print("Traced source-plane coordinates of `Grid2D` pixel 1:")
print(source_plane_grid.native[0, 1, :])

"""
__Ray Tracing Images__

We now have grid of coordinates in the source-plane.

This means we can compute how the source galaxy's light appears after gravitational lensing.

By passing the source-plane grid to the source galaxy's `image_2d_from` method, we can compute its lensed image.
"""
source_image = source_galaxy.image_2d_from(grid=source_plane_grid)

"""
To be certain the source image has been lensed, lets plot it.

We will use a galaxies plotter again, however like the previous times we have used a plotter we now pass it a
ray-traced source-plane grid, as opposed to a uniform image-plane grid.
"""
mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Lensed Source Image"))

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=al.Galaxies(galaxies=[source_galaxy]),
    grid=source_plane_grid,
    mat_plot_2d=mat_plot,
)
galaxies_plotter.figures_2d(image=True)

"""
The lensed source appears as a rather spectacular ring of light!

Why is it a ring? Well, consider that:

- The lens galaxy is centred at (0.0", 0.0").
- The source-galaxy is centred at (0.0", 0.0").
- The lens galaxy is a spherical mass profile.
- The source-galaxy ia a spherical light profile.

Given the perfect symmetry of the system, every ray-traced path the source's light takes around the lens galaxy is 
radially identical. 

Therefore, nothing else but a ring of light can form!

This is called an 'Einstein Ring' and its radius is called the 'Einstein Radius', which are both named after the man 
who famously used gravitational lensing to prove his theory of general relativity.

We can also plot the "plane-image" of the source, which shows its appearance before it is lensed by the mass profile,
something that we cannot actually observe.
"""
galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=[source_galaxy],
    grid=source_plane_grid,
)
galaxies_plotter.figures_2d(plane_image=True)

"""
__Ray Tracing Grids__

Lets inspect the image-plane grid and source-plane grid in more detail, using a `Grid2DPlotter`.
"""
mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Image-plane Grid"))

grid_plotter = aplt.Grid2DPlotter(grid=image_plane_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Source-plane Grid"))

grid_plotter = aplt.Grid2DPlotter(grid=source_plane_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
The source-plane gridlooks very interesting! 

We can see it is not regular, not uniform, and has an aestetically pleasing visual appearance. Remember that every 
coordinate on this source-plane grid (e.g. every black dot) corresponds to a coordinate on the image-plane grid that 
has been deflected by our mass profile; this is strong gravitational lensing in action!

We can zoom in on the central regions of the source-plane to reveal a 'diamond like' structure with a fractal like 
appearance.
"""
mat_plot = aplt.MatPlot2D(
    title=aplt.Title(label="Source-plane Grid2D Zoomed"),
    axis=aplt.Axis(extent=[-0.1, 0.1, -0.1, 0.1]),
)

grid_plotter = aplt.Grid2DPlotter(grid=source_plane_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Automatic Ray Tracing__

We manually performed ray-tracing above, to illustrate how the calculations are performed, but this is cumbersome and
time consuming.

Tracers have methods which perform the ray-tracing calculations we illustrated above for us. 

For example, after supplying our tracer with galaxies it is simply to compute an image of the entire strong lens system
using its `image_2d_from` method:
"""
traced_image_2d = tracer.image_2d_from(grid=image_plane_grid)
print("traced image pixel 1")
print(traced_image_2d.native[0, 0])
print("traced image pixel 2")
print(traced_image_2d.native[0, 1])
print("traced image pixel 3")
print(traced_image_2d.native[0, 2])

"""
This image appears as the Einstein ring we saw in the previous tutorial.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=image_plane_grid)
tracer_plotter.figures_2d(image=True)

"""
When this function is called, behind the scenes autolens is performing the following steps:

1) Use the lens's mass profiles to compute the deflection angle at every image-plane $(y,x)$ grid coordinate.
2) Subtract every deflection angles from its corresponding image-plane coordinate to compute the source-plane grid.
3) Use the source-plane galaxies to compute the light of the lensed source after ray tracing.

Above, we also inspect the image-plane grid and source-plane grid, which were computed manually.

The tracer's `traced_grid_2d_list_from` returns the traced grid of every plane:
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=image_plane_grid)

"""
The first traced grid corresponds to the image-plane grid (i.e. before lensing) which we plotted above:
"""
print("grid image-plane (y,x) coordinate 1")
print(traced_grid_list[0].native[0, 0])
print("grid image-plane (y,x) coordinate 2")
print(traced_grid_list[0].native[0, 1])


"""
The second grid is the source-plane grid, which we again plotted above and previously computed manually:
"""
print("grid source-plane (y,x) coordinate 1")
print(traced_grid_list[1].native[0, 0])
print("grid source-plane (y,x) coordinate 2")
print(traced_grid_list[1].native[0, 1])

"""
We can use the `TracerPlotter` to plot these planes and grids.
"""
include = aplt.Include2D(grid=True)

tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=image_plane_grid, include_2d=include
)
tracer_plotter.figures_2d_of_planes(plane_image=True, plane_grid=True, plane_index=0)
tracer_plotter.figures_2d_of_planes(plane_image=True, plane_grid=True, plane_index=1)

"""
__Log10 Space__

As discussed in previous tutorials, the light and mass profiles of galaxies are often better described in log10 space.

The same API can be used to make these plots for a `TracerPLotter` as used previously.

This works for any quantity that can be plotted, below we just use a `plane_image` as an example.
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=image_plane_grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
tracer_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
A ray-tracing subplot plots the following:

 1) The image, computed by ray-tracing the source-galaxy's light from the source-plane to the image-plane.
 2) The source-plane image, showing the source-galaxy's intrinsic appearance (i.e. if it were not lensed).
 3) The image-plane convergence, computed using the lens galaxy's total mass distribution.
 4) The image-plane gravitational potential, computed using the lens galaxy's total mass distribution.
 5) The image-plane deflection angles, computed using the lens galaxy's total mass distribution.
"""
tracer_plotter.subplot_tracer()

"""
Just like for a profile and galaxies, these quantities attributes can be computed via a `*_from` method.
"""
convergence_2d = tracer.convergence_2d_from(grid=image_plane_grid)

print("Tracer convergence at coordinate 1:")
print(convergence_2d.native[0, 0])
print("Tracer convergence at coordinate 2:")
print(convergence_2d.native[0, 1])
print("Tracer convergence at coordinate 101:")
print(convergence_2d.native[1, 0])

"""
The tracer convergence is identical the summed convergence of its lens galaxies.
"""
image_plane_convergence_2d = tracer.planes[0].convergence_2d_from(grid=image_plane_grid)

print("Image-Plane convergence at coordinate 1:")
print(image_plane_convergence_2d.native[0, 0])
print("Image-Plane convergence at coordinate 2:")
print(image_plane_convergence_2d.native[0, 1])
print("Image-Plane convergene at coordinate 101:")
print(image_plane_convergence_2d.native[1, 0])

"""
I've left the rest below commented to avoid too many print statements, but if you're feeling adventurous go ahead 
and uncomment the lines below!
"""
# print("Potential:")
# print(tracer.potential_2d_from(grid=image_plane_grid))
# print(tracer.image_plane.potential_2d_from(grid=image_plane_grid))
# print("Deflections:")
# print(tracer.deflections_yx_2d_from(grid=image_plane_grid))
# print(tracer.image_plane.deflections_yx_2d_from(grid=image_plane_grid))

"""
The `TracerPlotter` can also plot the above attributes as individual figures:
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=image_plane_grid)
tracer_plotter.figures_2d(
    image=True,
    convergence=True,
    potential=False,
    deflections_y=False,
    deflections_x=False,
)

"""
__Mappings__

Lets plot the image and source planes next to one another and highlight specific points on both. The coloring of the 
highlighted points therefore shows how specific image pixels **map** to the source-plane (and visa versa).

This is the first time we have used the `Visuals2D` object, which allows the appearance of **PyAutoLens** figures to 
be customized. We'll see this object crop up throughout the **HowToLens** lectures, and a full description of all
of its options is provided in the `autolens_workspace/plot` package.

Below, we input integer `indexes` that highlight the image-pixels that correspond to those indexes in 
a different color. We highlight indexes running from 0 -> 50, which appear over the top row of the image-plane grid,
alongside numerous other indexes.
"""
visuals = aplt.Visuals2D(
    indexes=[
        range(0, 50),
        range(500, 550),
        [1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250],
        [6250, 8550, 8450, 8350, 8250, 8150, 8050, 7950, 7850, 7750],
    ]
)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Image-plane Grid"))

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=tracer.planes[0],
    grid=image_plane_grid,
    mat_plot_2d=mat_plot,
    visuals_2d=visuals,
)
galaxies_plotter.figures_2d(plane_grid=True)

mat_plot = aplt.MatPlot2D(title=aplt.Title(label="Source-plane Grid"))

source_plane_grid = tracer.traced_grid_2d_list_from(grid=image_plane_grid)[1]

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=tracer.planes[1],
    grid=source_plane_grid,
    mat_plot_2d=mat_plot,
    visuals_2d=visuals,
)
galaxies_plotter.figures_2d(plane_grid=True)

"""
__Wrap Up__

We have finally performed actual strong lensing ray-tracing calculations! 

Now, its time for you to explore lensing phenomena in more detail. In particular, you should try:

 1) Changing the lens galaxy's einstein radius, what happens to the source-plane`s image?

 2) Change the lens's mass profile from a `IsothermalSph` to an `Isothermal`, making sure to input 
 `ell_comps` that are not (0.0, 0.0). What happens to the number of source images?

Try to make an the image-plane with two galaxies, both with mass profiles, and see how multi-galaxy lensing can 
produce extremely irregular images of a single source galaxy. Also try making a source-plane with multiple galaxies, 
and see how weird and irregular you can make the lensed image appear.
"""
