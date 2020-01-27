import autolens as al
import autolens.plot as aplt

# To begin chapter 4, we'll begin by learning about pixelizations, which we apply to a source-plane to reconstruct a
# source galaxy's light.

# Lets setup a lensed source-plane grid, using a lens galaxy and tracer (our source galaxy doesn't have a light profile,
# as we're going to reconstruct its light using a pixelization).
grid = al.grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[1]

# Next, lets set up a pixelization using the 'pixelizations' module, which we've imported as 'pix'.

# There are multiple pixelizations available in PyAutoLens. For now, we'll keep it simple and use a uniform
# rectangular grid. As usual, the grid's 'shape' defines its (y,x) dimensions.
rectangular = al.pix.Rectangular(shape=(25, 25))

# By itself, a pixelization doesn't tell us much. It has no grid of coordinates, no image, and nothing which tells it
# about the lens we're fitting. This information comes when we use the pixelization to set up a 'mapper'. We'll use
# the (traced) source-plane grid to set up this mapper.

mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)

# This mapper is a 'RectangularMapper' - every pixelization generates it owns mapper.
print(type(mapper))

# By plotting our mapper, we now see our pixelization. Its a fairly boring grid of rectangular pixels
# (we'll cover what the 'inversion' means in a later tutorial).
aplt.mapper_obj(
    mapper=mapper,
    include=aplt.Include(inversion_grid=True, inversion_pixelization_grid=True),
    plotter=aplt.Plotter(
        labels=aplt.Labels(title="Fairly Boring Grid of Rectangular Pixels")
    ),
)

# However, the mapper does contain lots of interesting information about our pixelization, for example it contains both
# the pixelization grid defining the rectangular grid's pixel centers...
print("Rectangular Grid Pixel Centre 1:")
print(mapper.pixelization_grid.in_2d[0, 0])
print("Rectangular Grid Pixel Centre 2:")
print(mapper.pixelization_grid.in_2d[0, 1])
print("Rectangular Grid Pixel Centre 3:")
print(mapper.pixelization_grid.in_2d[0, 2])
print("etc.")

# ... and the source-plane grid we used to setup the mapper. Crucially, because the mapper has *both* grids, it will
# be able to provide us with the mappings between them!
print("Source-Plane Grid Pixel Centre 1:")
print(mapper.grid.in_2d[0, 0])
print("Source-Plane Grid Pixel Centre 2:")
print(mapper.grid.in_2d[0, 1])
print("Source-Plane Grid Pixel Centre 3:")
print(mapper.grid.in_2d[0, 2])
print("etc.")

# Infact, we can plot the source-plane grid and rectangular pixelization grid on our pixelization - to make it look
# slightly less boring!
aplt.mapper_obj(
    mapper=mapper,
    include=aplt.Include(inversion_grid=True, inversion_pixelization_grid=True),
    plotter=aplt.Plotter(
        labels=aplt.Labels(title="Slightly less Boring Grid of Rectangular Pixels")
    ),
)

# Finally, the mapper has lots more information about the pixelizations, for example, the arc-second
# size and dimensions.
print(mapper.pixelization_grid.shape_2d_scaled)
print(mapper.pixelization_grid.scaled_maxima)
print(mapper.pixelization_grid.scaled_minima)

# And with that, we're done. This was a relatively gentle overview of pixelizations, but one that
# was hopefully easy to follow. Think about the following questions before moving on to the next tutorial:

# 1) Look at how the source-grid coordinates are distributed over the rectangular pixel-grid. Are these points
#    distributed evenly over the rectangular grid's pixels? Do some pixels have a lot more grid-points inside of them?
#    Do some pixels have no grid-points in them?

#  2) The rectangular pixelization's edges are aligned with the most exterior coordinates of the source-grid. This is
#     intentional - why do you think this is?
