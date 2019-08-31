import autofit as af
import autolens as al
import os

# This tool allows one to make simulated data-sets of strong lenses using a multi-plane ray-tracer, such that all
# galaxies down the line-of-sight are included in the ray-tracing calculation based on their redshifts.

# The 'data name' is the name of the data folder and 'data_name' the folder the data is stored in, e.g:

# The image will be output as '/workspace/data/data_type/data_name/image.fits'.
# The noise-map will be output as '/workspace/data/data_type/data_name/lens_name/noise_map.fits'.
# The psf will be output as '/workspace/data/data_type/data_name/psf.fits'.

# Setup the path to the workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# (these files are already in the workspace and are remade running this script)
data_type = "example"
data_name = "multi_plane"

# Create the path where the data will be output, which in this case is
# '/workspace/data/example/multi_plane/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

# The pixel scale of the image to be simulated
pixel_scale = 0.05

# Simulate a simple Gaussian PSF for the image.
psf = al.PSF.from_gaussian(shape=(11, 11), sigma=0.05, pixel_scale=pixel_scale)

# Setup the image-plane al.ogrid of the CCD array which will be used for generating the image of the
# simulated strong lens.
grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
    shape=(400, 400), pixel_scale=pixel_scale, sub_grid_size=1
)

# Setup the lens galaxy's light mass (SIE) and source galaxy light (elliptical Sersic) for this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
    shear=al.mass_profiles.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.light_profiles.EllipticalSersic(
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
    light=al.light_profiles.SphericalSersic(
        centre=(4.0, 4.0), intensity=0.30, effective_radius=0.3, sersic_index=2.0
    ),
    mass=al.mass_profiles.SphericalIsothermal(centre=(4.0, 4.0), einstein_radius=0.02),
)

los_1 = al.Galaxy(
    redshift=0.75,
    light=al.light_profiles.SphericalSersic(
        centre=(3.6, -5.3), intensity=0.20, effective_radius=0.6, sersic_index=1.5
    ),
    mass=al.mass_profiles.SphericalIsothermal(centre=(3.6, -5.3), einstein_radius=0.04),
)

los_2 = al.Galaxy(
    redshift=1.25,
    light=al.light_profiles.SphericalSersic(
        centre=(-3.1, -2.4), intensity=0.35, effective_radius=0.4, sersic_index=2.5
    ),
    mass=al.mass_profiles.SphericalIsothermal(
        centre=(-3.1, -2.4), einstein_radius=0.03
    ),
)

# Use these galaxies to setup a multi-plane tracer, which will generate the image for the simulated CCD
# instrument. This tracer orders galaxies by redshift and performs ray-tracing based on their line-of-sight redshifts.
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy, los_0, los_1, los_2]
)

# Lets look at the tracer's image - this is the image we'll be simulating.
al.ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer, grid=grid)

# Now lets simulate the CCD instrument, remembering that we use a special image which ensures edge-effects don't
# degrade our modeling of the telescope optics (e.g. the PSF convolution).
ccd = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

# Lets plot the simulated CCD instrument before we output it to files.
al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd)

# Finally, lets output our simulated instrument to the data path as .fits files.
al.output_ccd_data_to_fits(
    ccd_data=ccd,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)
