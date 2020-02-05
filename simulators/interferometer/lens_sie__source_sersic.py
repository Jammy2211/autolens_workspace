import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# This tool allows one to make simulated datasets of strong lenses, which can be used to test example pipelines and
# investigate strong lens modeling on simulated datasets where the 'true' answer is known.

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the dataset is stored in, e.g:

# The image will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/image.fits'.
# The noise-map will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/lens_name/noise_map.fits'.
# The psf will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/psf.fits'.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# (these files are already in the autolens_workspace and are remade running this script)
dataset_label = "interferometer"
dataset_name = "lens_sie__source_sersic"

# Create the path where the dataset will be output, which in this case is
# '/autolens_workspace/dataset/interferometer/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# The real space pixel scale of the interferometer data used to simulate it.
real_space_pixel_scales = 0.1

# To perform the Fourier transform we need the wavelengths of the baselines, which we'll load from the fits file below.
uv_wavelengths_path = workspace_path + "dataset/" + dataset_label + "/uv_wavelengths/"

uv_wavelengths = al.util.array.numpy_array_1d_from_fits(
    file_path=uv_wavelengths_path + "uv_wavelengths.fits", hdu=0
)

# To simulate the interferometer dataset we first create a simulator, which defines the shape, resolution and pixel-scale of the
# visibilities that are simulated, as well as its expoosure time, noise levels and uv-wavelengths.
simulator = al.simulator.interferometer(
    real_space_shape_2d=(151, 151),
    real_space_pixel_scales=real_space_pixel_scales,
    uv_wavelengths=uv_wavelengths,
    sub_size=4,
    exposure_time=300.0,
    background_level=0.1,
    noise_sigma=0.1,
)

# Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.9, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


# Use these galaxies to setup a tracer, which will generate the image for the simulated interferometer dataset.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# Lets look at the tracer's image - this is the image we'll be simulating.

# To make this figure, we need to pass the plotter a grid which it uses to create the image. The simulator has its
# grid accessible as a property, which we can use to do this.
aplt.tracer.profile_image(tracer=tracer, grid=simulator.grid)

# We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
# interferometer dataset.
interferometer = simulator.from_tracer(tracer=tracer)

# Lets plot the simulated interferometer dataset before we output it to fits.
aplt.interferometer.subplot_interferometer(interferometer=interferometer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
interferometer.output_to_fits(
    visibilities_path=dataset_path + "visibilities.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    uv_wavelengths_path=dataset_path + "uv_wavelengths.fits",
    overwrite=True,
)
