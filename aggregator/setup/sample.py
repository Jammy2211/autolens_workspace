import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# This script creates a sample of three strong lenses, which are used to illustrate the aggregator.

# It follows the scripts described in the '/autolens_workspace/simulators/', so if anything doesn't make sense check
# those scripts out for details!

# Setup the path to the autolens_workspace, using a relative directory name.
aggregator_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

######## EXAMPLE LENS SYSTEM 1 ###########

# (these files are already in the autolens_workspace and are remade running this script)
dataset_label = "aggregator_sample"
dataset_name = "lens_sie__source_sersic__0"

# Create the path where the dataset will be output.
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=aggregator_path, folder_names=["dataset", dataset_label, dataset_name]
)

# The pixel scale of the image to be simulated.
pixel_scales = 0.1

# Simulate a simple Gaussian PSF for the image.
psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=pixel_scales)

# Create a simulator, which defines the shape, resolution and pixel-scale of the image that is simulated, as well as
# its expoosure time, noise levels and psf.
simulator = al.simulator.imaging(
    shape_2d=(100, 100),
    pixel_scales=pixel_scales,
    sub_size=4,
    exposure_time=300.0,
    psf=psf,
    background_level=0.1,
    add_noise=True,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=0.8, axis_ratio=0.8, phi=0.0
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=0.0,
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.0,
    ),
)

# Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

aplt.tracer.profile_image(tracer=tracer, grid=simulator.grid)

# We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
# imaging dataset.
imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)


######## EXAMPLE LENS SYSTEM 2 ###########

# (these files are already in the autolens_workspace and are remade running this script)
dataset_label = "aggregator_sample"
dataset_name = "lens_sie__source_sersic__1"

# Create the path where the dataset will be output.
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=aggregator_path, folder_names=["dataset", dataset_label, dataset_name]
)

# The pixel scale of the image to be simulated.
pixel_scales = 0.1

# Simulate a simple Gaussian PSF for the image.
psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=pixel_scales)

# Create a simulator, which defines the shape, resolution and pixel-scale of the image that is simulated, as well as
# its expoosure time, noise levels and psf.
simulator = al.simulator.imaging(
    shape_2d=(100, 100),
    pixel_scales=pixel_scales,
    sub_size=4,
    exposure_time=300.0,
    psf=psf,
    background_level=0.1,
    add_noise=True,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.7, phi=45.0
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.2, 0.2),
        axis_ratio=0.7,
        phi=45.0,
        intensity=0.3,
        effective_radius=1.5,
        sersic_index=2.5,
    ),
)

# Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

aplt.tracer.profile_image(tracer=tracer, grid=simulator.grid)

# We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
# imaging dataset.
imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)


######## EXAMPLE LENS SYSTEM 3 ###########

# (these files are already in the autolens_workspace and are remade running this script)
dataset_label = "aggregator_sample"
dataset_name = "lens_sie__source_sersic__2"

# Create the path where the dataset will be output.
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=aggregator_path, folder_names=["dataset", dataset_label, dataset_name]
)

# The pixel scale of the image to be simulated.
pixel_scales = 0.1

# Simulate a simple Gaussian PSF for the image.
psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=pixel_scales)

# Create a simulator, which defines the shape, resolution and pixel-scale of the image that is simulated, as well as
# its expoosure time, noise levels and psf.
simulator = al.simulator.imaging(
    shape_2d=(100, 100),
    pixel_scales=pixel_scales,
    sub_size=4,
    exposure_time=300.0,
    psf=psf,
    background_level=0.1,
    add_noise=True,
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.2, axis_ratio=0.6, phi=90.0
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.3, 0.3),
        axis_ratio=0.6,
        phi=90.0,
        intensity=0.3,
        effective_radius=2.0,
        sersic_index=3.0,
    ),
)

# Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

aplt.tracer.profile_image(tracer=tracer, grid=simulator.grid)

# We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
# imaging dataset.
imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)
