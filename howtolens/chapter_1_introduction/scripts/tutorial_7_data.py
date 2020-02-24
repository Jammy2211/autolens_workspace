import autolens as al
import autolens.plot as aplt

# In this example, we'll use the 'imaging' module to 'simulator' imaging of a strong lens made
# using a tracer. By simulator, we mean that it will appear as if we had observed it using a real telescope,
# with this example making an image representative of Hubble Space Telescope imaging.

# To simulate an image, we need to model the telescope's optics. We'll do this by convolving the image with a
# Point-Spread Function, which we can simulator as a Gaussian using the abstract-simulator module.
psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=0.1)

# To simulator imaging data, we use a grid, like usual.
grid = al.grid.uniform(shape_2d=(100, 100), pixel_scales=0.1, sub_size=2)

# Now, lets setup our lens galaxy, source galaxy and tracer.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=45.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# Lets look at the tracer's image - this is the image we'll be simulating.
aplt.tracer.profile_image(tracer=tracer, grid=grid)

# To Simulate the imaging dataset, we don't use the image plotted above. Instead, we use an image which has been generated
# specifically for simulating an image, which pads the arrays it is computed on based on the shape of the PSF we
# convolve the image with. This ensures edge-effects do not degrade our dataset's PSF convolution.
normal_image = tracer.profile_image_from_grid(grid=grid)
padded_image = tracer.padded_profile_image_from_grid_and_psf_shape(
    grid=grid, psf_shape_2d=psf.shape_2d
)
print(normal_image.shape)
print(padded_image.shape)

# Now, to simulate the imaging dataset, we setup a 'simulator' and pass it the tracer, which adds the following effects
# to the image:

# 1) Telescope optics: Using the Point Spread Function above.
# 2) The Background Sky: Although the image that is returned is automatically background sky subtracted.
# 3) Poisson noise: Due to the background sky, lens galaxy and source galaxy Poisson photon counts.

simulator = al.simulator.imaging(
    shape_2d=grid.shape_2d,
    pixel_scales=0.1,
    sub_size=grid.sub_size,
    exposure_time=300.0,
    psf=psf,
    background_level=0.1,
    add_noise=True,
)

imaging = simulator.from_tracer(tracer=tracer)

# Lets plot the image - we can see the image has been blurred due to the telescope optics and noise has been added.
aplt.imaging.image(imaging=imaging)

# Finally, lets output these files to.fits files, we'll begin to analyze them in the next tutorial!
chapter_path = "/path/to/AutoLens/autolens_workspace/howtolens/chapter_1_introduction"
chapter_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace/howtolens/chapter_1_introduction/"

# The dataset path specifies where the dataset is output, this time in the directory 'chapter_path/dataset'
dataset_path = chapter_path + "dataset/"

# Now output our simulated dataset to hard-disk.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    psf_path=dataset_path + "psf.fits",
    overwrite=True,
)
