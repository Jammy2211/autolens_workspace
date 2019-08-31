import autolens as al
from autolens.model.inversion import pixelizations as pix
from autolens.model.inversion.plotters import mapper_plotters

# To begin chapter 4, we'll begin by learning about pixelizations, which we apply to a source-plane to reconstruct a
# source galaxy's light.

# Lets setup a lensed source-plane grid, using a lens galaxy and tracer (our source galaxy doesn't have a light profile,
# as we're going to reconstruct its light using a pixelization).
grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
    shape=(100, 100), pixel_scale=0.05, sub_grid_size=2
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[1]

# Next, lets set up a pixelization using the 'pixelizations' module, which we've imported as 'pix'.

# There are multiple pixelizations available in PyAutoLens. For now, we'll keep it simple and use a uniform
# rectangular grid. As usual, the grid's 'shape' defines its (y,x) dimensions.
rectangular = pix.Rectangular(shape=(25, 25))

# By itself, a pixelization doesn't tell us much. It has no grid of coordinates, no image, and nothing which tells it
# about the lens we're fitting. This information comes when we use the pixelization to set up a 'mapper'. We'll use
# the (traced) source-plane grid to set up this mapper.

mapper = rectangular.mapper_from_grid_and_pixelization_grid(grid=source_plane_grid)

# This mapper is a 'RectangularMapper' - every pixelization generates it owns mapper.
print(type(mapper))

# By plotting our mapper, we now see our pixelization. Its a fairly boring grid of rectangular pixels.
mapper_plotters.plot_rectangular_mapper(
    mapper=mapper,
    should_plot_grid=False,
    title="Fairly Boring Grid of Rectangular Pixels",
)

# However, the mapper does contain lots of interesting information about our pixelization, for example its geometry
# attribute tells us where the pixel centers are located.
print("Rectangular Grid Pixel Centre 1:")
print(mapper.geometry.pixel_centres[0])
print("Rectangular Grid Pixel Centre 2:")
print(mapper.geometry.pixel_centres[1])
print("Rectangular Grid Pixel Centre 3:")
print(mapper.geometry.pixel_centres[2])
print("etc.")

# Infact, we can plot these centre on our grid - to make it look slightly less boring!
mapper_plotters.plot_rectangular_mapper(
    mapper=mapper, should_plot_grid=False, should_plot_centres=True
)

# The mapper also has the (source-plane) grid that we passed when we set it up. Lets check they're the same.
print("Source Grid Pixel 1")
print(source_plane_grid[0])
print(mapper.grid[0])
print("Source Grid Pixel 2")
print(source_plane_grid[1])
print(mapper.grid[1])
print("etc.")

# We can over-lay the grid on top. Its starting to look a bit less boring now!
mapper_plotters.plot_rectangular_mapper(
    mapper=mapper, should_plot_centres=True, should_plot_grid=True
)


# Finally, the mapper and its geometry has lots more information about the pixelization, for example, the arc-second
# size and dimensions.
print(mapper.geometry.shape_arcsec)
print(mapper.geometry.arc_second_maxima)
print(mapper.geometry.arc_second_minima)

# And with that, we're done. This was a relatively gentle overview of pixelizations, but one that
# was hopefully easy to follow. Think about the following questions before moving on to the next tutorial:

# 1) Look at how the source-grid coordinates are distributed over the rectangular pixel-grid. Are these points
#    distributed evenly over the rectangular grid's pixels? Do some pixels have a lot more grid-points inside of them?
#    Do some pixels have no grid-points in them?

#  2) The rectangular pixelization's edges are aligned with the most exterior coordinates of the source-grid. This is
#     intentional - why do you think this is?
