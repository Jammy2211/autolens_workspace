"""
Tutorial 2: Ray Tracing
=======================

Strong gravitational lensing occurs when the mass of a foreground galaxy (or galaxies) curves space-time around it,
causing light rays from a background source to appear deflected.

The process of ray tracing calculates how much the path of each light ray is bent by the mass of the foreground galaxy.
It then traces the paths of these light rays back to the observer, allowing us to determine how the source appears
distorted. As a result, the images of the source may appear as arcs, multiple images, or other complex patterns.

In the previous tutorial, we introduced **light profiles**, which are analytic functions that describe the distribution
of light from a galaxy. In this tutorial, we will focus on **mass profiles**, which are analytic functions that
describe the mass distribution within a galaxy. A key quantity derived from these profiles is the **deflection angle**,
which quantifies how light is deflected by the mass of the galaxy at any point in space.

Grids were crucial in the previous tutorial, as they enabled us to compute the light profile of a galaxy at every
coordinate point. In ray tracing, grids are equally important because they are used to calculate the deflection
angles of light rays caused by a mass profile and to map the source's light rays back to the observer.

Let’s revisit the schematic of strong lensing:

![Schematic of Gravitational Lensing](https://i.imgur.com/zB6tIdI.jpg)

As observers, we do not see the true appearance of the source (e.g., a round blob of light). Instead, we only
perceive its light after it has been deflected and lensed by the foreground galaxies. This resulting image is known
as the **observed image** or **image-plane image**, which we will produce through the ray-tracing process.

The schematic above uses the terms "image-plane" and "source-plane." In the context of gravitational lensing, a "plane"
refers to a collection of galaxies located at the same redshift, meaning they are physically aligned parallel to one
another.

In this tutorial, we will create a strong lensing system consisting of planes similar to the one depicted above. While
a plane can contain multiple galaxies, we will focus on a simple scenario with just one lens galaxy and one source
galaxy.

Here is an overview of what we'll cover in this tutorial:

- **Grid**: How the 2D grids that were important for evaluating light profiles are equally important
  for ray-tracing calculations.

- **Mass Profiles**: Introduce mass profiles, which describe the mass distribution of galaxies and are used to
  calculate deflection angles.

- **Ray Tracing Grids**: How to map the light rays from the image-plane to the source-plane using the
 deflection angles and **lens equation**.

- **Ray Tracing Images**: How to evaluate the lensed image of a source galaxy after it has been
  gravitationally lensed.

- **Galaxies**: How to include both light and mass profiles in a single `Galaxy` object, and therefore
construct realistic lens and source galaxies.

- **Tracer**: Introduce the `Tracer` object, which automates the ray-tracing process and allows us to compute
  images of the entire lens system.

- **Mappings**: Visualize how image pixels map to the source plane and vice versa using the `Visuals2D` object.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt


"""
__Grid__

In the previous tutorial, we created 2D grids of (y,x) coordinates and showed how shifting and rotating these grids 
is crucial for evaluating the light profiles of galaxies.

Grids are also essential for performing ray-tracing calculations. The coordinates in the grid are deflected by the 
lens galaxy, and these deflected coordinates are used in ray-tracing calculations.

Now, let’s create the grid for this tutorial, which we’ll call the `image_plane_grid`. It represents the grid of 
coordinates in the image plane, before the light is deflected by the lens galaxy. This grid is uniform, meaning 
every coordinate is evenly spaced. However, this uniformity will change after ray-tracing, as the light rays are 
mapped to the source plane.
"""
image_plane_grid = al.Grid2D.uniform(shape_native=(101, 101), pixel_scales=0.1)

"""
__Mass Profiles__

To perform lensing calculations, we use mass profiles available in the `mass_profile` module (accessible via `al.mp`).

A mass profile is an analytic function that describes the mass distribution within a galaxy. It is used to calculate 
deflection angles and other quantities like surface density and gravitational potential.

In gravitational lensing, deflection angles describe how a mass bends light by curving space-time.

We will start with a simple mass profile, `IsothermalSph`, which represents a spherically symmetric isothermal 
mass distribution. This profile has two main properties: its center (the origin of the coordinate system) and its 
Einstein radius (which indicates the galaxy's mass and how much it bends light rays).
"""
sis_mass_profile = al.mp.IsothermalSph(
    centre=(0.0, 0.0),  # The (y,x) arc-second coordinates of the profile's center.
    einstein_radius=1.6,  # The Einstein radius of the profile in arc-seconds.
)
print(sis_mass_profile)

"""
In the previous tutorial, we used the `image_2d_from` method to compute the image of a light profile by evaluating 
its intensity at each (y,x) coordinate on the grid.

Mass profiles have a similar method called `deflections_yx_2d_from`, which calculates the deflection angles at 
every (y,x) coordinate on the grid in units of arc-seconds.
"""
deflections = sis_mass_profile.deflections_yx_2d_from(grid=image_plane_grid)

"""
Like grids and arrays, the deflection angles can be accessed using the `native` and `slim` attributes. These are 
structured similarly to a `Grid2D` object:

- **native**: A 2D array with shape \([total_y_pixels, total_x_pixels, 2]\), where the last dimension represents the 
  (y,x) deflection components.
  
- **slim**: A 1D array with shape \([total_y_pixels * total_x_pixels, 2]\), where the coordinates are flattened 
  into a single list.
"""
print("Deflection angles of pixel 0:")
print(deflections.native[0, 0])
print("Deflection angles of pixel 1:")
print(deflections.slim[1])

"""
There is an important difference between a grid and deflection angles. A `Grid2D` is a set of coordinates, while 
deflection angles are 2D vectors. This means that each deflection angle is defined at a specific (y,x) coordinate 
but has two components: a y and an x value, which vary across the grid.

This is why the method is called `deflections_yx_2d_from`—the `yx` signifies that these are 2D vectors with both 
y and x components.

The deflection angles are stored in a `VectorYX2D` data structure:
"""
print(type(deflections))

"""
This structure includes a `grid`, which represents the `Grid2D` of coordinates where the deflection angles are 
calculated (in this case, the `image_plane_grid` we defined earlier). It also has vector-specific methods, 
such as `magnitude`, which calculates the magnitude of each deflection vector using \((x^2 + y^2)^{0.5}\).
"""
print("Deflection angle's `Grid2D` at pixel 0:")
print(deflections.grid.native[0, 0])
print("Deflection angle magnitude at pixel 0:")
print(deflections.magnitudes.native[0, 0])

"""
We can use a `MassProfilePlotter` to visualize the deflection angles, which displays the y and x components separately. 

On this plot, you’ll see yellow and white lines called **critical curves**. These curves are important in lensing 
and will be explained in detail in the next tutorial.
"""
mass_profile_plottter = aplt.MassProfilePlotter(
    mass_profile=sis_mass_profile, grid=image_plane_grid
)
mass_profile_plottter.figures_2d(deflections_y=True, deflections_x=True)

"""
Mass profiles also have additional properties used in lensing calculations:

- **convergence**: Represents the surface mass density of the profile in dimensionless units.
- **potential**: Represents the "lensing potential" of the mass profile in dimensionless units.
- **magnification**: Indicates how much brighter light rays appear due to the focusing effect of lensing.

These quantities can be calculated using `*_from` methods and are returned as `Array2D` objects.
"""
convergence = sis_mass_profile.convergence_2d_from(grid=image_plane_grid)
potential_2d = sis_mass_profile.potential_2d_from(grid=image_plane_grid)
magnification_2d = sis_mass_profile.magnification_2d_from(grid=image_plane_grid)

"""
The same plotter API can be used to visualize these properties:
"""
mass_profile_plottter.figures_2d(convergence=True, potential=True, magnification=True)

"""
One-dimensional plots are also available, showing how these quantities change radially from the center of the mass profile:
"""
mass_profile_plottter.figures_1d(convergence=True, potential=True)

"""
The **convergence** and **potential** can be better understood when plotted in logarithmic space:
"""
mass_profile_plottter = aplt.MassProfilePlotter(
    mass_profile=sis_mass_profile,
    grid=image_plane_grid,
    mat_plot_2d=aplt.MatPlot2D(use_log10=True),
)
mass_profile_plottter.figures_2d(convergence=True, potential=True)

"""
While the **convergence** and **potential** are fundamental to lensing, their detailed physical meaning is not 
essential for learning ray-tracing calculations, so we will cover them more thoroughly in a later tutorial.

__Ray Tracing Grids__

We are now ready to use the deflection angles to perform ray tracing. 

We therefore ray-trace every 2d (y,x) coordinate $\theta$ from the image-plane to its (y,x)  source-plane 
coordinate $\beta$ using the deflection angles $\alpha$ of the mass profile

 $\beta = \theta - \alpha(\theta)$
 
The equation above is one of the most important in lensing -- it is called the **lens equation**. It describes how
light rays are deflected by mass from the image-plane to the source-plane.

By subtracting the $(y,x)$ deflection angles values from the $(y,x)$ grid of image-plane coordinates we can therefore 
determine the `source_plane_grid`.
"""
source_plane_grid = image_plane_grid - deflections

print("Ray traced source-plane coordinates of `Grid2D` pixel 0:")
print(source_plane_grid.native[0, 0, :])
print("Ray traced source-plane coordinates of `Grid2D` pixel 1:")
print(source_plane_grid.native[0, 1, :])
"""
Let’s compare the image-plane and source-plane grids to observe how the mass profile has deflected the coordinates 
from the image-plane to the source-plane.

Several key differences emerge:

- The **image-plane grid** is uniform, while the **source-plane grid** is not. This is because the mass profile has 
deflected the image-plane coordinates, making them no longer evenly spaced.

- The **source-plane grid** is much smaller than the image-plane grid. This shrinking occurs because the mass profile 
magnifies the light rays, compressing the source-plane coordinates into a smaller region within the image-plane grid.

- The **source-plane grid** forms a central diamond-shaped structure known as the **caustic**. This structure has s
everal important properties in gravitational lensing that will be covered in later tutorials.
"""
grid_plotter = aplt.Grid2DPlotter(grid=image_plane_grid)
grid_plotter.set_title("Image-plane Grid")
grid_plotter.figure_2d()

grid_plotter = aplt.Grid2DPlotter(grid=source_plane_grid)
grid_plotter.set_title("Source-plane Grid")
grid_plotter.figure_2d()

"""
We can zoom in on the central regions of the source-plane to better observe the diamond-shaped structure, which 
has a fractal-like appearance and is known as the **caustic**.
"""
mat_plot = aplt.MatPlot2D(
    title=aplt.Title(label="Source-plane Zoomed"),
    axis=aplt.Axis(extent=[-0.1, 0.1, -0.1, 0.1]),
)

grid_plotter = aplt.Grid2DPlotter(grid=source_plane_grid, mat_plot_2d=mat_plot)
grid_plotter.figure_2d()

"""
__Ray Tracing Images__

With a grid of coordinates in the source-plane, we can now evaluate a light profile in this plane and see how its 
light appears after being gravitationally lensed.

By passing the source-plane grid to the `image_2d_from` method of a light profile, we can create the image of the 
light profile after lensing.
"""
sersic_light_profile = al.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.1),
    intensity=1.0,
    effective_radius=2.0,
    sersic_index=1.2,
)

print(source_plane_grid)

# Ignore the use of a `Grid2DIrregular` here and input into an `Array2D`, these are some annoying API quirks
# that will be fixed in the future.

lensed_source_image = sersic_light_profile.image_2d_from(
    grid=al.Grid2DIrregular(
        values=source_plane_grid,
    )
)

lensed_source_image = al.Array2D(
    values=lensed_source_image, mask=source_plane_grid.mask
)

array_plotter = aplt.Array2DPlotter(array=lensed_source_image)
array_plotter.set_title("Lensed Source Image")
array_plotter.figure_2d()

"""
The resulting lensed source appears as a striking ring of light!

Why does it form a ring? Consider the following:

- The lens galaxy is centered at (0.0", 0.0").
- The source galaxy is centered at (0.0", 0.0").
- The lens galaxy has a spherical mass profile.
- The source galaxy has a spherical light profile.

Due to the perfect symmetry of this system, every ray-traced path the light from the source takes around the lens 
galaxy is identical. As a result, the light forms a **ring**.

This ring is known as an **Einstein Ring**, named after the scientist who famously used gravitational lensing to 
confirm his theory of general relativity. The radius of this ring is called the **Einstein Radius**.

We can also plot the source’s appearance in the source-plane, before it is lensed by the mass profile. This is known
as the **plane-image** of the source. It is computed by passing a uniform grid to the `image_2d_from` method of the
light profile. This grid can have whatever resolution we choose, allowing us to clearly see the source's shape.

**Exercise**: Try changing the lens galaxy's `einstein radius`, what happens to the source-plane`s image?
"""
grid = al.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.02)

source_image = sersic_light_profile.image_2d_from(grid=grid)

array_plotter = aplt.Array2DPlotter(array=source_image)
array_plotter.set_title("Unlensed Source Image")
array_plotter.figure_2d()

"""
__Galaxies__

In previous tutorials, we used the `Galaxy` object to represent galaxies with multiple light profiles. We saw how 
these objects included methods similar to those of light profiles, such as `image_2d_from`, which sums the images of 
the individual light profiles to produce an image of the entire galaxy.

When adding mass profiles to `Galaxy` objects, the behavior is analogous. For example, the 
galaxy's `deflections_yx_2d_from` method calculates the deflection angles for all mass profiles within the galaxy,
which are then summed to produce the total deflection angle.

We can also include both light and mass profiles in a single `Galaxy` object, which is crucial for representing the 
foreground lens galaxy that we observe and whose mass causes the lensing effect.

Below, we create two galaxies: a **lens galaxy** and a **source galaxy**, reflecting their roles in the lensing process.

The galaxies' redshifts are now more significant, as they determine the sequence of calculations in ray tracing. For 
instance, the lens galaxy must be at a lower redshift than the source galaxy, as light rays are deflected in the 
image-plane before tracing to the source-plane.
"""
sis_mass_profile = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.SersicCore(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.1),
        intensity=1.0,
        effective_radius=2.0,
        sersic_index=1.5,
    ),
    mass=sis_mass_profile,
)

print(lens_galaxy)

sersic_light_profile = al.lp.SersicSph(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=2.0, sersic_index=1.5
)

source_galaxy = al.Galaxy(redshift=1.0, light=sersic_light_profile)

print(source_galaxy)

"""
We can use the `GalaxyPlotter` object to visualize the image and deflection angles of the lens galaxy:
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=lens_galaxy, grid=image_plane_grid)
galaxy_plotter.figures_2d(image=True, deflections_y=True, deflections_x=True)

"""
By passing the `source_plane_grid` to the `GalaxyPlotter` for the source galaxy, we can visualize the lensed 
appearance of the source galaxy:
"""
galaxy_plotter = aplt.GalaxyPlotter(galaxy=source_galaxy, grid=source_plane_grid)
galaxy_plotter.figures_2d(image=True)

"""
__Tracer__

The ray-tracing calculations above were performed manually to demonstrate the process. However, in practice, 
manually computing the deflection angles of each galaxy and managing grids for forming lensed images is cumbersome.

The `Tracer` object simplifies this process and is a central tool in **PyAutoLens**. 

When galaxies are provided to the `Tracer`, it performs the following tasks:

1. The galaxies are sorted in order of increasing redshift.
2. The galaxies are grouped into "planes" based on their unique redshift values.

The `Tracer` then automates the calculations for quantities like images, deflection angles, and potentials. 
It performs the ray-tracing calculations for each plane and combines the results to obtain the final quantities.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

"""
The `Tracer` object contains a `planes` attribute, where each plane is a group of galaxies at the same redshift. 

In this example, with two galaxies at different redshifts, the tracer has two planes. The first plane corresponds 
to the image-plane, and the second to the source-plane.
"""
print(tracer.planes)
print(len(tracer.planes))

"""
We can inspect each plane, which shows the galaxies within them:
"""
print("Image Plane:")
print(tracer.planes[0])
print()
print("Source Plane:")
print(tracer.planes[1])

"""
This structure allows us to perform calculations for each plane separately. For example, we can compute and plot the 
deflection angles for all galaxies in the image-plane:
"""
deflections_yx_2d = tracer.planes[0].deflections_yx_2d_from(grid=image_plane_grid)
print("Deflection angles of the image plane at pixel 0:")
print(deflections_yx_2d.native[0, 0, 0])
print(deflections_yx_2d.native[0, 0, 1])

"""
We can also use the `GalaxiesPlotter` to visualize the deflection angles for all galaxies within a plane:
"""
galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=tracer.planes[0], grid=image_plane_grid
)
galaxies_plotter.figures_2d(deflections_y=True, deflections_x=True)

"""
The `Tracer` object has an `image_2d_from` method that computes the image of the entire lens system, accounting 
for the ray-tracing across planes. 

In this example, the `Tracer` sums the images from two profiles—one from the lens galaxy and one from the source 
galaxy. The lens galaxy’s image appears unaltered since it lies in the image-plane before any ray-tracing occurs, 
while the source galaxy's image is lensed as it is in the source-plane.
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=image_plane_grid)
tracer_plotter.figures_2d(image=True)

"""
The `image_2d_from` method of the `Tracer` returns the combined image, which includes the lens galaxy's light and the 
lensed image of the source galaxy:
"""
traced_image_2d = tracer.image_2d_from(grid=image_plane_grid)
print("Traced image pixel 1:")
print(traced_image_2d.native[0, 0])

"""
Just like with light and mass profiles of galaxies, these quantities can be visualized in log10 space using 
the `TracerPlotter`:
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=image_plane_grid, mat_plot_2d=aplt.MatPlot2D(use_log10=True)
)
tracer_plotter.figures_2d_of_planes(plane_index=1, plane_image=True)

"""
When the `image_2d_from` method is called, the following steps occur:

1. The lens galaxy's mass profiles are used to calculate the deflection angles at each image-plane $(y,x)$ coordinate.
2. These deflection angles are subtracted from the corresponding image-plane coordinates, forming the source-plane grid
   using the lens equation.
3. The source-plane galaxies' profiles are used to compute the lensed light after ray tracing.

These are the same steps we manually performed earlier in this tutorial!

**Exercise**: Change the lens's mass profile from a `IsothermalSph` to an elliptical `Isothermal`, making sure to 
input  `ell_comps` that are not (0.0, 0.0). What happens to the number of source images?

The `traced_grid_2d_list_from` method of the `Tracer` returns the traced grids for each plane:
"""
traced_grid_list = tracer.traced_grid_2d_list_from(grid=image_plane_grid)

"""
The first grid corresponds to the original image-plane grid, while the second is the deflected source-plane grid:
"""
print("Image-plane (y,x) coordinate 1:")
print(traced_grid_list[0].native[0, 0])
print("Source-plane (y,x) coordinate 1:")
print(traced_grid_list[1].native[0, 0])

"""
These planes and grids can be visualized using the `TracerPlotter`:
"""
include = aplt.Include2D(grid=True)
tracer_plotter = aplt.TracerPlotter(
    tracer=tracer, grid=image_plane_grid, include_2d=include
)
tracer_plotter.figures_2d_of_planes(plane_image=True, plane_grid=True, plane_index=0)
tracer_plotter.figures_2d_of_planes(plane_image=True, plane_grid=True, plane_index=1)

"""
A ray-tracing subplot shows the following:

1. The lensed image, created by tracing the source-galaxy's light from the source-plane to the image-plane.
2. The source-plane image, revealing the source-galaxy's intrinsic shape (if it were not lensed).
3. The image-plane convergence, based on the lens galaxy's mass.
4. The image-plane gravitational potential.
5. The image-plane deflection angles.
6. The image-plane magnification.
"""
tracer_plotter.subplot_tracer()

"""
The tracer also provides methods to compute various quantities, like convergence:
"""
convergence = tracer.convergence_2d_from(grid=image_plane_grid)
print("Tracer convergence at coordinate 1:")
print(convergence.native[0, 0])

"""
This convergence is identical to the combined convergence of the lens galaxies:
"""
convergence = tracer.planes[0].convergence_2d_from(grid=image_plane_grid)
print("Image-Plane convergence at coordinate 1:")
print(convergence.native[0, 0])

"""
The `TracerPlotter` can also plot other attributes, like deflection angles, as individual figures:
"""
tracer_plotter.figures_2d(
    image=True,
    convergence=True,
    potential=False,
    deflections_y=False,
    deflections_x=False,
)

"""
__Mappings__

Let’s plot the image and source planes side by side and highlight specific points on each. This allows us to 
visualize how certain image pixels **map** to the source plane, and vice versa, based on their coloring.

This is the first time we are using the `Visuals2D` object, which enables customization of the appearance 
of **PyAutoLens** figures. We'll encounter this object throughout the **HowToLens** tutorials. A detailed 
explanation of all its options is available in the `autolens_workspace/plot` package.

Below, we specify integer `indexes` that highlight specific image pixels with different colors. We highlight 
indexes ranging from 0 to 50, which correspond to pixels across the top row of the image-plane grid, along with 
several other specific indexes.

"""
visuals = aplt.Visuals2D(
    indexes=[
        range(0, 50),
        range(500, 550),
        [1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250],
        [6250, 8550, 8450, 8350, 8250, 8150, 8050, 7950, 7850, 7750],
    ]
)

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=tracer.planes[0],
    grid=image_plane_grid,
    visuals_2d=visuals,
)
galaxies_plotter.set_title("Image-plane Grid")
galaxies_plotter.figures_2d(plane_grid=True)

"""
Next, we the source-plane grid and highlight specific points on it:
"""
source_plane_grid = tracer.traced_grid_2d_list_from(grid=image_plane_grid)[1]

galaxies_plotter = aplt.GalaxiesPlotter(
    galaxies=tracer.planes[1],
    grid=source_plane_grid,
    visuals_2d=visuals,
)
galaxies_plotter.set_title("Source-plane Grid")
galaxies_plotter.figures_2d(plane_grid=True)

"""
__Wrap Up__

You've learn how strong lens ray-tracing works and how to do it using the `Tracer` object in **PyAutoLens**.

Lets summarize what we've learnt:

- **Grid**: The grid of coordinates that was important for evaluating light profiles is equally important for
  ray-tracing calculations. 
  
- **Mass Profiles**: These profiles describe the mass distribution of galaxies and are used to calculate deflection
  angles.
  
- **Ray Tracing Grids**: By subtracting the deflection angles from the image-plane grid, we can map the light rays
  to the source-plane and plot grid coordinates in both planes.
  
- **Ray Tracing Images**: We can evaluate the lensed image of a source galaxy using a ray-traced grid in order
  to show how it appears after being gravitationally lensed.

- **Tracer**: The `Tracer` object automates the ray-tracing process, allowing us to compute images of the entire 
  lens system.
 
- **Mappings**: We can visualize how image pixels map to the source plane and vice versa using the `Visuals2D` 
  object.
  
**Exercise:** Experiment with making very complex `Tracer`'s with multiple galaxies at different redshifts,
and where these galaxies have many different light and mass profiles. How complex can you make the gravitational
lensing configuration? Do you think such complex configurations are common in the Universe?
"""
