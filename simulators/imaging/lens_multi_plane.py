import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# This tool allows one to make simulated datasets of strong lenses using a multi-plane ray-tracer, such that all
# galaxies down the line-of-sight are included in the ray-tracing calculation based on their redshifts.

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the dataset is stored in, e.g:

# The image will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/image.fits'.
# The noise-map will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/lens_name/noise_map.fits'.
# The psf will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/psf.fits'.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# (these files are already in the autolens_workspace and are remade running this script)
dataset_label = "imaging"
dataset_name = "lens_multi_plane"

# Create the path where the dataset will be output, which in this case is
# '/autolens_workspace/dataset/imaging/multi_plane/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# The pixel scale of the image to be simulated
pixel_scales = 0.05

# Simulate a simple Gaussian PSF for the image.
psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=pixel_scales)

# To simulate the imaging dataset we first create a simulator, which defines the shape, resolution and pixel-scale of the
# image that is simulated, as well as its expoosure time, noise levels and psf.
simulator = al.simulator.imaging(
    shape_2d=(100, 100),
    pixel_scales=pixel_scales,
    sub_size=4,
    exposure_time=300.0,
    psf=psf,
    background_level=0.1,
    add_noise=True,
)

# Setup the lens galaxy's light mass (SIE) and source galaxy light (elliptical Sersic) for this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        axis_ratio=0.8,
        phi=60.0,
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

# Setup our line-of-sight (los) galaxies using Spherical Sersic profiles for their light and Singular
# Isothermal Sphere (SIS) profiles. We'll use 3 galaxies, but you can add more if desired.
los_0 = al.Galaxy(
    redshift=0.25,
    light=al.lp.SphericalSersic(
        centre=(4.0, 4.0), intensity=0.30, effective_radius=0.3, sersic_index=2.0
    ),
    mass=al.mp.SphericalIsothermal(centre=(4.0, 4.0), einstein_radius=0.02),
)

los_1 = al.Galaxy(
    redshift=0.75,
    light=al.lp.SphericalSersic(
        centre=(3.6, -5.3), intensity=0.20, effective_radius=0.6, sersic_index=1.5
    ),
    mass=al.mp.SphericalIsothermal(centre=(3.6, -5.3), einstein_radius=0.04),
)

los_2 = al.Galaxy(
    redshift=1.25,
    light=al.lp.SphericalSersic(
        centre=(-3.1, -2.4), intensity=0.35, effective_radius=0.4, sersic_index=2.5
    ),
    mass=al.mp.SphericalIsothermal(centre=(-3.1, -2.4), einstein_radius=0.03),
)

# Use these galaxies to setup a multi-plane tracer, which will generate the image for the simulated Imaging
# dataset. This tracer orders galaxies by redshift and performs ray-tracing based on their line-of-sight redshifts.
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy, los_0, los_1, los_2]
)

# To make this figure, we need to pass the plotter a grid which it uses to create the image. The simulator has its
# grid accessible as a property, which we can use to do this.
aplt.tracer.profile_image(tracer=tracer, grid=simulator.grid)

# We can then pass this simulator a tracer, which uses the tracer to create a ray-traced image which is simulated as
# imaging dataset following the setup of the dataset.
imaging = simulator.from_tracer(tracer=tracer)

# Lets plot the simulated Imaging dataset before we output it to fits.
aplt.imaging.subplot_imaging(imaging=imaging)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)
