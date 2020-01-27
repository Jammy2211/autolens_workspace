import autolens as al
import autolens.plot as aplt

# In the previous example, we made a mapper from a rectangular pixelization. However, it wasn't clear what a mapper
# was actually mapping. Infact, it didn't do much mapping at all! Therefore, in this tutorial, we'll cover mapping.

# To begin, lets simulate and load an image - it'll be clear why we're doing this in a moment.
def simulate():

    psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.7,
            phi=135.0,
            intensity=0.2,
            effective_radius=0.2,
            sersic_index=2.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(150, 150),
        pixel_scales=0.05,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


# Lets simulate our imaging dataset.
imaging = simulate()
aplt.imaging.subplot_imaging(imaging=imaging)

# Now, lets set up our grids (using the image above).
grid = al.grid.uniform(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1
)

# Our tracer will use the same lens galaxy and source galaxy that we used to simulate the imaging dataset(although, becuase
# we're modeling the source with a pixel-grid, we don't need to supply its light profile).
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.6
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=grid)[1]

# Next, we setup our pixelization and mapper using the tracer's source-plane grid.
rectangular = al.pix.Rectangular(shape=(25, 25))

mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)

# We're going to plot our mapper alongside the image we used to generate the source-plane grid.
aplt.mapper.subplot_image_and_mapper(
    image=imaging.image, mapper=mapper, include=aplt.Include(inversion_grid=True)
)

# The pixels in the image map to the pixels in the source-plane, and visa-versa. Lets highlight a set of
# image-pixels in both the image and source-plane.
aplt.mapper.subplot_image_and_mapper(
    image=imaging.image,
    mapper=mapper,
    include=aplt.Include(inversion_grid=True),
    image_pixel_indexes=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    ],
)

# That's nice, and we can see the mappings, but it isn't really what we want to know, is it? We really want to go the
# other way, and see how our source-pixels map to the image. This is where mappers come into their own, as they let us
# map all the points in a given source-pixel back to the image. Lets map source pixel 313, the central
# source-pixel, to the image.
aplt.mapper.subplot_image_and_mapper(
    image=imaging.image,
    mapper=mapper,
    include=aplt.Include(inversion_grid=True),
    source_pixel_indexes=[[312]],
)

# And there we have it - multiple imaging in all its glory. Try changing the source-pixel indexes of the line below.
# This will give you a feel for how different regions of the source-plane map to the image.
aplt.mapper.subplot_image_and_mapper(
    image=imaging.image,
    mapper=mapper,
    include=aplt.Include(inversion_grid=True),
    source_pixel_indexes=[[312, 318], [412]],
)

# Okay, so I think we can agree, mappers map things! More specifically, they map our source-plane pixels to pixels
# in the observed image of a strong lens.

# Finally, lets do the same as above, but using a masked image. By applying a mask, the mapper will only map
# image-pixels inside the mask. This removes the (many) image pixels at the edge of the image, where the source
# isn't present. These pixels also pad-out the source-plane, thus by removing them our source-plane reduces in size.
# Lets just have a quick look at these edges pixels:
aplt.mapper.subplot_image_and_mapper(
    image=imaging.image,
    mapper=mapper,
    include=aplt.Include(inversion_grid=True),
    source_pixel_indexes=[[0, 1, 2, 3, 4, 5, 6, 7], [620, 621, 622, 623, 624]],
)

# Lets use an annular mask, which will capture the ring-like shape of the lensed source galaxy.
mask = al.mask.circular_annular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    sub_size=1,
    inner_radius=1.0,
    outer_radius=2.2,
)

# Lets quickly confirm the annuli capture the source's light.
aplt.imaging.image(imaging=imaging, mask=mask)

# As usual, we setup our image and mask up as lens dataset and create a tracer using its (now masked) al.
masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, al.Galaxy(redshift=1.0)])

source_plane_grid = tracer.traced_grids_of_planes_from_grid(grid=masked_imaging.grid)[1]

# Finally, we use the masked source-plane grid to setup a new mapper (using the same rectangular 25 x 25
# pixelization as before).
mapper = rectangular.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)

# Lets have another look.
aplt.mapper.subplot_image_and_mapper(
    image=masked_imaging.image,
    mapper=mapper,
    include=aplt.Include(mask=True, inversion_grid=True),
)

# Woah! Look how much closer we are to the source-plane (The axis sizes have decreased from ~ -2.5" -> 2.5" to
# ~ -0.6" to 0.6"). We can now really see the diamond of points in the centre of the source-plane (for those who have
# been reading up, this diamond is called the 'caustic').
aplt.mapper.subplot_image_and_mapper(
    image=masked_imaging.image,
    mapper=mapper,
    include=aplt.Include(mask=True, inversion_grid=True),
    source_pixel_indexes=[[312], [314], [316], [318]],
)

# Great - tutorial 2 down! We've learnt about mappers, which map things, and we used them to understand how the image
# and source plane map to one another. Your exercises are:

# 1) Change the einstein radius of the lens galaxy in small increments (e.g. einstein radius 1.6" -> 1.55").
#    As the radius deviates from 1.6" (the input value of the simulated lens), what do you notice about where the
#    points map from the centre of the source-plane (where the source-galaxy is simulated, e.g. (0.0", 0.0"))?

# 2) Incrementally increase the axis ratio of the lens's mass profile to 1.0. What happens to quadruple imaging?

# 3) Now, finally, think - how is all of this going to help us actually model lenses? We've said we're going to
#    reconstruct our source galaxies on the pixel-grid. So, how does knowing how each pixel maps to the image
#    actually help us? If you've not got any bright ideas, then worry not - that exactly what we're going to cover
#    in the next tutorial.
