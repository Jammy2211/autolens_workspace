"""
Example: Input Source Image
===========================

This example illustrates how to create a lensed image of a source, from a image of a source (E.g. the image is
discrete pixel intensity values on a square or rectangular grid).

Typically the source image will be a high resolution unlensed galaxy image, in order to simulate strong lenses
with realistic source emission.

However, it could be an image of anything, so you could make a lensed image of your dog if you really wanted!
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
from scipy.interpolate import griddata
import autolens as al
import autolens.plot as aplt

"""
__Galaxy Image__

We first load the image of the galaxy (from a .fits file) which will be lensed.

This image is typically a real galaxy image that is not gravitationally lensed. 
"""
data_path = path.join("scripts", "misc", "galaxy_image.fits")

galaxy_image = al.Array2D.from_fits(file_path=data_path, pixel_scales=0.02)

"""
To create the lensed image, we will ray-trace image pixels to the source-plane and interpolate them onto the 
source galaxy image
"""
source_plane_grid = al.Grid2D.uniform(
    shape_native=galaxy_image.shape_native,
    pixel_scales=galaxy_image.pixel_scales,
    origin=(0.0, 0.0),
)

"""
__Image-PLane Grid__

The 2D grid of (y,x) coordinates which we will ray-trace to the source-plane (via a lens model) and compare to the
source-galaxy image pixel fluxes to create our lensed image.
"""
image_plane_grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

"""
__Tracer__

Define the mass model which is used for the ray-tracing, from which the lensed source image will be created.

An input `source` galaxy is required below, so that the `Tracer` has a source-plane (at redshift 1.0)  which the
image-plane grid's coordinates are ray-traced too.
"""
lens = al.Galaxy(
    redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.0)
)
source = al.Galaxy(redshift=1.0)

tracer = al.Tracer(galaxies=[lens, source])

"""
Ray-trace the image-plane grid to the source-plane.

This is the grid we will overlay the source image, in order to created the lensed source image.
"""
traced_image_plane_grid = tracer.traced_grid_2d_list_from(grid=image_plane_grid)[-1]

"""
__Interpolation__

We now use the scipy interpolation function `griddata`, where:

 - `points`: the 2D grid of (y,x) coordinates representing the location of every pixel of the galaxy image from
 which we are creating the lensed source image.
 
 - `values`: the intensity values of the galaxy image which is being used to create the lensed source image.
 
 - `xi`: the image-plane grid ray traced to the source-plane, defining the image on which the lensed source is created.
 
The interpolation works by pairing every ray-traced (y,x) coordinate in the `traced_image_plane_grid` to its
closest 4 coordinates in `source_plane_grid`. 

It then uses Delaunay interpolation to compute the intensity from these 4 coordinates.

"""
lensed_image = griddata(
    points=source_plane_grid, values=galaxy_image, xi=traced_image_plane_grid
)

"""
__Lensed Source Image__

We can plot the lensed source image to make sure it looks sensible.
"""
lensed_image = al.Array2D.no_mask(
    values=lensed_image,
    shape_native=image_plane_grid.shape_native,
    pixel_scales=image_plane_grid.pixel_scales,
)

array_2d_plotter = aplt.Array2DPlotter(array=lensed_image)
array_2d_plotter.figure_2d()
