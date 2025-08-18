"""
Tutorial 2: Mappers
===================

In the previous tutorial, we used a pixelization to create made a `Mapper`. However, it was not clear what a `Mapper`
does, why it was called a mapper and whether it was mapping anything at all!

Therefore, in this tutorial, we'll cover mappers in more detail.

WARNING: THHIS TUTORIAL VISUALS ARE SLIGHTLY BUGGY CURRENTLY AND WILL BE FIXED IN THE FUTURE.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

we'll use new strong lensing data, where:

 - The lens galaxy's light is omitted.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    psf_path=dataset_path / "psf.fits",
    pixel_scales=0.1,
    over_sample_size_pixelization=1,
)

"""
Now, lets set up our `Grid2D` (using the image above).
"""
grid = al.Grid2D.uniform(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    over_sample_size=1,
)

"""
Our `Tracer` will use the same lens galaxy and source galaxy that we used to Simulate the imaging data (although, 
becuase we're modeling the source with a pixel-grid, we do not pass it any light profiles.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

tracer = al.Tracer(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grid_2d_list_from(grid=grid)[1]

"""
__Mappers__

We now setup a `Pixelization` and use it to create a `Mapper` via the tracer`s source-plane grid, just like we did in
the previous tutorial.
"""
mesh = al.mesh.Rectangular(shape=(25, 25))

pixelization = al.Pixelization(mesh=mesh)

mapper_grids = pixelization.mapper_grids_from(
    mask=grid.mask,
    source_plane_data_grid=source_plane_grid,
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

"""
We first plot the mapper's pixelization, which is a grid of rectangular pixels in the source-plane.

We plot this next to the image using the `subplot_image_and_mapper` method.
"""
mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
)
mapper_plotter.subplot_image_and_mapper(
    image=dataset.data,
)

"""
Using the `Visuals2D` object we are also going to highlight specific grid coordinates certain colors, such that we
can see how they map from the image-plane to source-plane and visa versa.

We do this by specifying their integer indexes, corresponding to the index of each data point in the image and source
plane grids. These indexes are used to highlight the grid coordinates in the image and source-plane grids that map
directly to one another.
"""
indexes = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
]

visuals = aplt.Visuals2D(
    indexes=indexes,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)
mapper_plotter.subplot_image_and_mapper(
    image=dataset.data,
)

"""
It can help to plot the image-plane grid and source-plane grid over each image, in order to see how the
mapped pixels relate to the image and source-plane grids overall.

This requires us to plot each image individually (e.g. not using `subplot_image_and_mapper`), so we can plot each
grid via the visuals object and have them displayed clearly.
"""
visuals = aplt.Visuals2D(
    grid=mapper_grids.image_plane_data_grid,
    indexes=indexes,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.figure_2d_image(image=dataset.data)

visuals = aplt.Visuals2D(
    grid=mapper_grids.source_plane_data_grid,
    indexes=indexes,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.figure_2d()

"""
We can now make these mappings appear the other way round. That is, we can input a source-pixel index (of our 
rectangular grid) and highlight how all of the (sub-)image-pixels that it contains map to the image-plane. 

To make the indexes appear in the image-plane, we have to convert them from their source-plane pixel indexes
to image plane image-pixel indexes using the mapper.

Lets map source pixel 312, the central source-pixel, to the image sub-grid and then plot it via the visuals object.

The pixels, plotted in red, extended beyond the central square pixel of the source-plane grid. This is because
the pairing of data pixels to source pixels is not one-to-one, as an interpolation scheme is used to map pixels
which land near the edges of the source-pixel, but outside them, to that source pixel with a weight.
"""
pix_indexes = [[312]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(indexes=indexes)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
There we have it, multiple imaging in all its glory. 

Try changing the source-pixel indexes of the line below. This will give you a feel for how different regions of the 
source-plane map to the image.
"""
pix_indexes = [[312, 318], [412]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(
    indexes=indexes,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
Okay, so I think we can agree, mapper's map things! More specifically, they map source-plane pixels to multiple pixels 
in the observed image of a strong lens.

__Mask__

Finally, lets repeat the steps that we performed above, but now using a masked image. By applying a `Mask2D`, the 
mapper only maps image-pixels that are not removed by the mask. This removes the (many) image pixels at the edge of the 
image, where the source is not present. These pixels also pad-out the source-plane, thus by removing them our 
source-plane reduces in size.

Lets just have a quick look at these edges pixels:
"""
pix_indexes = [[0, 1, 2, 3, 4, 5, 6, 7], [620, 621, 622, 623, 624]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(
    indexes=indexes,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
Lets use an annular `Mask2D`, which will capture the ring-like shape of the lensed source galaxy.
"""
mask = al.Mask2D.circular_annular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    inner_radius=1.0,
    outer_radius=2.2,
)

dataset = dataset.apply_mask(mask=mask)
dataset_plotter = aplt.ImagingPlotter(dataset=dataset, visuals_2d=visuals)
dataset_plotter.figures_2d(data=True)

"""
To create the mapper, we need to set up the masked imaging's grid as the source-plane gird via the tracer.
"""
tracer = al.Tracer(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grid_2d_list_from(grid=dataset.grids.pixelization)[1]

"""
We can now use the masked source-plane grid to create a new `Mapper` (using the same rectangular 25 x 25 pixelization 
as before).
"""
mapper_grids = mesh.mapper_grids_from(
    mask=mask, source_plane_data_grid=source_plane_grid
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

"""
Lets plot it, including the image-plane grid and source-plane grid, which are now much smaller than before because
the mask has removed the many image pixels at the edge of the image.
"""
visuals = aplt.Visuals2D(
    grid=mapper_grids.image_plane_data_grid,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.figure_2d_image(image=dataset.data)

visuals = aplt.Visuals2D(
    grid=mapper_grids.source_plane_data_grid,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.figure_2d()

"""
First, look how much closer we are to the source-plane (The axis sizes have decreased from ~ -2.5" -> 2.5" to 
~ -0.6" to 0.6"). 

We can more clearly see the diamond of points in the centre of the source-plane (for those who have been reading up, 
this diamond is called the `caustic`).

Now lets plot some indexes mapping from the source-plane to the image-plane, which will highlight how the
source-pixels map to the image pixels.
"""
pix_indexes = [[312], [314], [316], [318]]

indexes = mapper.slim_indexes_for_pix_indexes(pix_indexes=pix_indexes)

visuals = aplt.Visuals2D(
    indexes=indexes,
)

mapper_plotter = aplt.MapperPlotter(
    mapper=mapper,
    visuals_2d=visuals,
)

mapper_plotter.subplot_image_and_mapper(image=dataset.data)

"""
__Wrap Up__

In this tutorial, we learnt about mappers, and we used them to understand how the image and source plane map to one 
another. Your exercises are:

 1) Change the einstein radius of the lens galaxy in small increments (e.g. einstein radius 1.6" -> 1.55"). As the 
 radius deviates from 1.6" (the input value of the simulated lens), what do you notice about where the points map 
 from the centre of the source-plane (where the source-galaxy is simulated, e.g. (0.0", 0.0"))?
        
 2) Think about how this could help us actually model lenses. We have said we're going to reconstruct our source 
 galaxies on the pixel-grid. So, how does knowing how each pixel maps to the image actually help us? If you`ve not got 
 any bright ideas, then worry not, that exactly what we're going to cover in the next tutorial.
"""
