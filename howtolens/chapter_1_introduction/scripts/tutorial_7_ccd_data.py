from autolens.data.instrument import abstract_data
from autolens.data.instrument import ccd
from autolens.array import grids
from autolens.lens import ray_tracing
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens.plotters import ray_tracing_plotters
from autolens.data.plotters import ccd_plotters

# In this example, we'll use the 'ccd' module to 'simulate' ccd imaging of a strong lens made
# using a tracer. By simulate, we mean that it will appear as if we had observed it using a real telescope,
# with this example making an image representative of Hubble Space Telescope imaging.

# To simulate an image, we need to model the telescope's optics. We'll do this by convolving the image with a
# Point-Spread Function, which we can simulate as a Gaussian using the abstract-data module.
psf = abstract_data.PSF.from_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=0.1)

# To simulate ccd instrument, we use a grid, like usual.
grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
    shape=(100, 100), pixel_scale=0.1, sub_grid_size=2
)

# Now, lets setup our lens galaxy, source galaxy and tracer.
lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)

source_galaxy = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = ray_tracing.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# Lets look at the tracer's image - this is the image we'll be simulating.
ray_tracing_plotters.plot_profile_image(tracer=tracer, grid=grid)

# To Simulate the CCD data, we don't use the image plotted above. Instead, we use an image which has been generated
# specifically for simulating an image, which pads the array it is computed on based on the shape of the PSF we
# convolve the image with. This ensures edge-effects do not degrade our simulation's PSF convolution.
normal_image = tracer.profile_image_from_grid(grid=grid, return_in_2d=True)
padded_image = tracer.padded_profile_image_2d_from_grid_and_psf_shape(
    grid=grid, psf_shape=psf.shape
)
print(normal_image.shape)
print(padded_image.shape)

# Now, to simulate the ccd imaging data, we pass the tracer and grid to the ccd module's simulate
# function. This adds the following effects to the image:

# 1) Telescope optics: Using the Point Spread Function above.
# 2) The Background Sky: Although the image that is returned is automatically background sky subtracted.
# 3) Poisson noise: Due to the background sky, lens galaxy and source galaxy Poisson photon counts.

ccd_data = ccd.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=0.1,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

# Lets plot the image - we can see the image has been blurred due to the telescope optics and noise has been added.
ccd_plotters.plot_image(ccd_data=ccd_data)

# Finally, lets output these files to.fits files, we'll begin to analyze them in the next tutorial!
chapter_path = "/path/to/AutoLens/workspace/howtolens/chapter_1_introduction"
chapter_path = (
    "/home/jammy/PycharmProjects/PyAutoLens/workspace/howtolens/chapter_1_introduction/"
)

# The data path specifies where the data is output, this time in the directory 'chapter_path/data'
data_path = chapter_path + "data/"

# Now output our simulated data to hard-disk.
ccd.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    noise_map_path=data_path + "noise_map.fits",
    psf_path=data_path + "psf.fits",
    overwrite=True,
)
