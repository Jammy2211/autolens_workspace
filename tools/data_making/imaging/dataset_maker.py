import autofit as af
import autolens as al

import os

# This tool allows one to make simulated datasets of strong lenses, which can be used to test example pipelines and
# investigate strong lens modeling on simulated datasets where the 'true' answer is known.

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the dataset is stored in, e.g:

# The image will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/image.fits'.
# The noise-map will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/lens_name/noise_map.fits'.
# The psf will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/psf.fits'.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# (these files are already in the autolens_workspace and are remade running this script)
dataset_label = "imaging"
dataset_name = "lens_sersic_sie__source_sersic"

# Create the path where the dataset will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# The pixel scale of the image to be simulated
pixel_scales = 0.1

# Simulate a simple Gaussian PSF for the image.
psf = al.kernel.from_gaussian(shape_2d=(11, 11), sigma=0.1, pixel_scales=pixel_scales)

# To simulate the imaging dataset we first create a simulator, which defines the shape, resolution and pixel-scale of the
# image that is simulated, as well as its expoosure time, noise levels and psf.
simulator = al.simulator.imaging(
    shape_2d=(100, 100),
    pixel_scales=pixel_scales,
    sub_size=4,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

# Setup the lens galaxy's light (elliptical Sersic), mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = al.galaxy(
    redshift=0.5,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.galaxy(
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


# Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset.
tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# Lets look at the tracer's image - this is the image we'll be simulating.

# To make this figure, we need to pass the plotter a grid which it uses to create the image. The simulator has its
# grid accessible as a property, which we can use to do this.
al.plot.tracer.profile_image(tracer=tracer, grid=simulator.grid)

# We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
# imaging dataset-set.
imaging = simulator.from_tracer(tracer=tracer)

# Lets plot the simulated imaging dataset before we output it to fits.
al.plot.imaging.subplot(imaging=imaging)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)


# ####################################### OTHER EXAMPLE IMAGES #####################################

dataset_label = "imaging"
dataset_name = "lens_sersic_sie__source_sersic__2"


dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

lens_galaxy = al.galaxy(
    redshift=0.5,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.7,
        phi=80.0,
        intensity=0.8,
        effective_radius=1.3,
        sersic_index=2.5,
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.3, axis_ratio=0.6, phi=30.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.02, phi=145.0),
)

source_galaxy = al.galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(-0.2, -0.3),
        axis_ratio=0.9,
        phi=10.0,
        intensity=0.2,
        effective_radius=1.5,
        sersic_index=2.0,
    ),
)

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)

####################################### OTHER EXAMPLE IMAGES #####################################

dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

lens_galaxy = al.galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.galaxy(
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

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)

####################################### OTHER EXAMPLE IMAGES #####################################

dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic__2"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

lens_galaxy = al.galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.3, axis_ratio=0.8, phi=60.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.3, -0.4),
        axis_ratio=0.6,
        phi=40.0,
        intensity=0.2,
        effective_radius=1.2,
        sersic_index=2.0,
    ),
)

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)

####################################################

dataset_label = "imaging"
dataset_name = "lens_mass__source_sersic_x2"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

lens_galaxy = al.galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.9, phi=90.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy_0 = al.galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.25, 0.15),
        axis_ratio=0.7,
        phi=120.0,
        intensity=0.7,
        effective_radius=0.7,
        sersic_index=1.0,
    ),
)

source_galaxy_1 = al.galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.7, -0.5),
        axis_ratio=0.9,
        phi=60.0,
        intensity=0.2,
        effective_radius=1.6,
        sersic_index=3.0,
    ),
)

tracer = al.tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
)

imaging = simulator.from_tracer(tracer=tracer)

imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)

####################################################

dataset_label = "imaging"
dataset_name = "lens_bulge_disk_sie__source_sersic"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

lens_galaxy = al.galaxy(
    redshift=0.5,
    bulge=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=0.3,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.lp.EllipticalExponential(
        centre=(0.0, 0.0), axis_ratio=0.7, phi=30.0, intensity=0.2, effective_radius=1.6
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.galaxy(
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

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

imaging = simulator.from_tracer(tracer=tracer)

# Lets plot the simulated imaging dataset before we output it to fits.
al.plot.imaging.subplot(imaging=imaging)

imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)


####################################### OTHER EXAMPLE IMAGES #####################################

dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic__intervening_objects"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

lens_galaxy = al.galaxy(
    redshift=0.5,
    intervene_0=al.lp.SphericalExponential(
        centre=(1.0, 3.5), intensity=0.8, effective_radius=0.5
    ),
    intervene_1=al.lp.SphericalExponential(
        centre=(-2.0, -3.5), intensity=0.5, effective_radius=0.8
    ),
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.galaxy(
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

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)


####################################################

dataset_label = "imaging"
dataset_name = "lens_sie__subhalo_nfw__source_sersic"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

simulator = al.simulator.imaging(
    shape_2d=(150, 150),
    pixel_scales=0.05,
    sub_size=2,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

lens_galaxy = al.galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    subhalo=al.mp.SphericalTruncatedNFWMassToConcentration(
        centre=(1.6, 0.0), mass_at_200=1.0e10
    ),
    shear=al.mp.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.01, 0.01),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.3,
        effective_radius=0.3,
        sersic_index=2.5,
    ),
)

tracer = al.tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

imaging = simulator.from_tracer(tracer=tracer)

# Finally, lets output our simulated dataset to the dataset path as .fits files.
imaging.output_to_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    overwrite=True,
)
