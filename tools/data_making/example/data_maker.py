import autofit as af
import autolens as al

import os

# This tool allows one to make simulated data-sets of strong lenses, which can be used to test example pipelines and
# investigate strong lens modeling on data-sets where the 'true' answer is known.

# The 'data name' is the name of the data folder and 'data_name' the folder the data is stored in, e.g:

# The image will be output as '/workspace/data/data_type/data_name/image.fits'.
# The noise-map will be output as '/workspace/data/data_type/data_name/lens_name/noise_map.fits'.
# The psf will be output as '/workspace/data/data_type/data_name/psf.fits'.

# Setup the path to the workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# (these files are already in the workspace and are remade running this script)
data_type = "example"
data_name = "lens_sersic_sie__source_sersic"

# Create the path where the data will be output, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

# The pixel scale of the image to be simulated
pixel_scale = 0.1

# Simulate a simple Gaussian PSF for the image.
psf = al.PSF.from_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=pixel_scale)

# Setup the image-plane al.ogrid of the CCD array which will be used for generating the image of the
# simulated strong lens. The sub-grid size of 20x20 ensures we fully resolve the central regions of the lens and source
# galaxy light.
grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
    shape=(100, 100), pixel_scale=pixel_scale, sub_grid_size=16
)

# Setup the lens galaxy's light (elliptical Sersic), mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.light_profiles.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
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
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)


# Use these galaxies to setup a tracer, which will generate the image for the simulated CCD instrument.
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# Lets look at the tracer's image - this is the image we'll be simulating.
al.ray_tracing_plotters.plot_profile_image(tracer=tracer, grid=grid)

# Simulate the CCD instrument, remembering that we use a special image which ensures edge-effects don't
# degrade our modeling of the telescope optics (e.g. the PSF convolution).
ccd_data = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

# Lets plot the simulated CCD instrument before we output it to files.
al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

# Finally, lets output our simulated instrument to the data path as .fits files.
al.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)


# ####################################### OTHER EXAMPLE IMAGES #####################################

data_type = "example"
data_name = "lens_sersic_sie__source_sersic__2"


data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.light_profiles.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.7,
        phi=80.0,
        intensity=0.8,
        effective_radius=1.3,
        sersic_index=2.5,
    ),
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.3, axis_ratio=0.6, phi=30.0
    ),
    shear=al.mass_profiles.ExternalShear(magnitude=0.02, phi=145.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.light_profiles.EllipticalSersic(
        centre=(-0.2, -0.3),
        axis_ratio=0.9,
        phi=10.0,
        intensity=0.2,
        effective_radius=1.5,
        sersic_index=2.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

ccd_data = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)


# Finally, lets output our simulated instrument to the data path as .fits files.
al.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)

####################################### OTHER EXAMPLE IMAGES #####################################

data_type = "example"
data_name = "lens_sie__source_sersic"

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

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
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

ccd_data = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)


# Finally, lets output our simulated instrument to the data path as .fits files.
al.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)

####################################### OTHER EXAMPLE IMAGES #####################################

data_type = "example"
data_name = "lens_sie__source_sersic__2"

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.3, axis_ratio=0.8, phi=60.0
    ),
    shear=al.mass_profiles.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.light_profiles.EllipticalSersic(
        centre=(0.3, -0.4),
        axis_ratio=0.6,
        phi=40.0,
        intensity=0.2,
        effective_radius=1.2,
        sersic_index=2.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

ccd_data = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)


# Finally, lets output our simulated instrument to the data path as .fits files.
al.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)

####################################################

data_type = "example"
data_name = "lens_mass__source_sersic_x2"

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.9, phi=90.0
    ),
    shear=al.mass_profiles.ExternalShear(magnitude=0.05, phi=90.0),
)

source_galaxy_0 = al.Galaxy(
    redshift=1.0,
    light=al.light_profiles.EllipticalSersic(
        centre=(0.25, 0.15),
        axis_ratio=0.7,
        phi=120.0,
        intensity=0.7,
        effective_radius=0.7,
        sersic_index=1.0,
    ),
)

source_galaxy_1 = al.Galaxy(
    redshift=1.0,
    light=al.light_profiles.EllipticalSersic(
        centre=(0.7, -0.5),
        axis_ratio=0.9,
        phi=60.0,
        intensity=0.2,
        effective_radius=1.6,
        sersic_index=3.0,
    ),
)

tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy_0, source_galaxy_1]
)

ccd_data = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

al.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)

####################################################

data_type = "example"
data_name = "lens_bulge_disk_sie__source_sersic"

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.light_profiles.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=0.3,
        effective_radius=0.6,
        sersic_index=3.0,
    ),
    disk=al.light_profiles.EllipticalExponential(
        centre=(0.0, 0.0), axis_ratio=0.7, phi=30.0, intensity=0.2, effective_radius=1.6
    ),
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
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

ccd_data = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

# Lets plot the simulated CCD instrument before we output it to files.
al.ccd_plotters.plot_ccd_subplot(ccd_data=ccd_data)

al.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)


####################################### OTHER EXAMPLE IMAGES #####################################

data_type = "example"
data_name = "lens_sie__source_sersic__intervening_objects"

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["data", data_type, data_name]
)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    intervene_0=al.light_profiles.SphericalExponential(
        centre=(1.0, 3.5), intensity=0.8, effective_radius=0.5
    ),
    intervene_1=al.light_profiles.SphericalExponential(
        centre=(-2.0, -3.5), intensity=0.5, effective_radius=0.8
    ),
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
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

ccd_data = al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
    tracer=tracer,
    grid=grid,
    pixel_scale=pixel_scale,
    exposure_time=300.0,
    psf=psf,
    background_sky_level=0.1,
    add_noise=True,
)

# Finally, lets output our simulated instrument to the data path as .fits files.
al.output_ccd_data_to_fits(
    ccd_data=ccd_data,
    image_path=data_path + "image.fits",
    psf_path=data_path + "psf.fits",
    noise_map_path=data_path + "noise_map.fits",
    overwrite=True,
)
