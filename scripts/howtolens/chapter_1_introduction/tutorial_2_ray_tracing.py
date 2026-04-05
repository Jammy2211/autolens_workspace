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

- **Mappings**: Visualize how image pixels map to the source plane and vice versa using the `lines=`/`positions=` overlays object.

__Contents__

**Grid:** In the previous tutorial, we created 2D grids of (y,x) coordinates and showed how shifting and.
**Mass Profiles:** To perform lensing calculations, we use mass profiles available in the `mass_profile` module.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import autolens as al
import autoarray as aa
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
We can use `aplt.plot_array` to visualize the deflection angles, which displays the y and x components separately. 

On this plot, you’ll see yellow and white lines called **critical curves**. These curves are important in lensing 
and will be explained in detail in the next tutorial.
"""
deflections = sis_mass_profile.deflections_yx_2d_from(grid=image_plane_grid)
deflections_y = aa.Array2D(values=deflections.slim[:, 0], mask=image_plane_grid.mask)
aplt.plot_array(array=deflections_y, title="Deflections Y")
deflections = sis_mass_profile.deflections_yx_2d_from(grid=image_plane_grid)
deflections_x = aa.Array2D(values=deflections.slim[:, 1], mask=image_plane_grid.mask)
aplt.plot_array(array=deflections_x, title="Deflections X")

"""
Mass profiles also have additional properties used in lensing calculations:

- **convergence**: Represents the surface mass density of the profile in dimensionless units.
- **potential**: Represents the "lensing potential" of the mass profile in dimensionless units.
- **magnification**: Indicates how much brighter light rays appear due to the focusing effect of lensing.

These quantities can be calculated using `*_from` methods and are returned as `Array2D` objects.
"""
convergence = sis_mass_profile.convergence_2d_from(grid=image_plane_grid)
potential_2d = sis_mass_profile.potential_2d_from(grid=image_plane_grid)
magnification_2d = al.LensCalc.from_mass_obj(
    mass_obj=sis_mass_profile
).magnification_2d_from(grid=image_plane_grid)

"""
The same plotter API can be used to visualize these properties:
"""
aplt.plot_array(
    array=sis_mass_profile.convergence_2d_from(grid=image_plane_grid),
    title="Convergence",
)
aplt.plot_array(
    array=sis_mass_profile.potential_2d_from(grid=image_plane_grid), title="Potential"
)

"""
One-dimensional plots can also be made using the same projection technique as in the previous tutorial:
"""
grid_2d_projected = image_plane_grid.grid_2d_radial_projected_from(
    centre=sis_mass_profile.centre, angle=sis_mass_profile.angle()
)

convergence_1d = sis_mass_profile.convergence_2d_from(grid=grid_2d_projected)

plt.plot(grid_2d_projected[:, 1], convergence_1d)
plt.xlabel("Radius (arcseconds)")
plt.ylabel("Luminosity")
plt.show()
plt.close()

"""
The **convergence** and **potential** can be better understood when plotted in logarithmic space:
"""
