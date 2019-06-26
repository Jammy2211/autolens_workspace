import autofit as af
from autolens.data import ccd
from autolens.data import simulated_ccd
from autolens.data.array import grids
from autolens.lens import ray_tracing
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.lens.plotters import ray_tracing_plotters
from autolens.data.plotters import ccd_plotters

import os

# This tool allows one to make simulated data-sets of strong lenses, which can be used to test example pipelines and
# investigate strong lens modeling on data-sets where the 'true' answer is known.

# The 'data name' is the name of the data folder and 'data_name' the folder the data is stored in, e.g:

# The image will be output as '/workspace/data/data_type/data_name/image.fits'.
# The noise-map will be output as '/workspace/data/data_type/data_name/lens_name/noise_map.fits'.
# The psf will be output as '/workspace/data/data_type/data_name/psf.fits'.

# Setup the path to the workspace, using a relative directory name.
workspace_path = '{}/../../../'.format(os.path.dirname(os.path.realpath(__file__)))

# (these files are already in the workspace and are remade running this script)
data_type = 'example'
data_name = 'lens_light_mass_and_x1_source'

# Create the path where the data will be output, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

# The pixel scale of the image to be simulated
pixel_scale = 0.1

# Simulate a simple Gaussian PSF for the image.
psf = ccd.PSF.from_gaussian(
    shape=(11, 11), sigma=0.1, pixel_scale=pixel_scale)

# Setup the image-plane grid stack of the CCD array which will be used for generating the image-plane image of the
# simulated strong lens. The sub-grid size of 20x20 ensures we fully resolve the central regions of the lens and source
# galaxy light.
image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
    shape=(100, 100), pixel_scale=pixel_scale, psf_shape=psf.shape, sub_grid_size=16)

# Setup the lens galaxy's light (elliptical Sersic), mass (SIE+Shear) and source galaxy light (elliptical Sersic) for
# this simulated lens.
lens_galaxy = g.Galaxy(
    redshift=0.5,
    light=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=1.0, effective_radius=0.8,
                              sersic_index=4.0),
    mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0),
    shear=mp.ExternalShear(magnitude=0.05, phi=90.0))

source_galaxy = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0, intensity=0.3, effective_radius=1.0,
                              sersic_index=2.5))


# Use these galaxies to setup a tracer, which will generate the image-plane image for the simulated CCD data.
tracer = ray_tracing.TracerImageSourcePlanes(
    lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
    image_plane_grid_stack=image_plane_grid_stack)

# Lets look at the tracer's image-plane image - this is the image we'll be simulating.
ray_tracing_plotters.plot_image_plane_image(
    tracer=tracer)

# Simulate the CCD data, remembering that we use a special image-plane image which ensures edge-effects don't
# degrade our modeling of the telescope optics (e.g. the PSF convolution).
simulated_ccd = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
    image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=pixel_scale,
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets plot the simulated CCD data before we output it to files.
ccd_plotters.plot_ccd_subplot(ccd_data=simulated_ccd)

# Finally, lets output our simulated data to the data path as .fits files.
ccd.output_ccd_data_to_fits(
    ccd_data=simulated_ccd,
    image_path=data_path + 'image.fits',
    psf_path=data_path + 'psf.fits',
    noise_map_path=data_path + 'noise_map.fits',
    overwrite=True)


####################################### OTHER EXAMPLE IMAGES #####################################

data_type = 'example'
data_name = 'lens_mass_and_x1_source'

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0),
    shear=mp.ExternalShear(magnitude=0.05, phi=90.0))

source_galaxy = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0, intensity=0.3, effective_radius=1.0,
                              sersic_index=2.5))

tracer = ray_tracing.TracerImageSourcePlanes(
    lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
    image_plane_grid_stack=image_plane_grid_stack)

simulated_ccd = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
    image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=pixel_scale,
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)


# Finally, lets output our simulated data to the data path as .fits files.
ccd.output_ccd_data_to_fits(
    ccd_data=simulated_ccd,
    image_path=data_path + 'image.fits',
    psf_path=data_path + 'psf.fits',
    noise_map_path=data_path + 'noise_map.fits',
    overwrite=True)

####################################################

data_type = 'example'
data_name = 'lens_mass_and_x2_source'

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

lens_galaxy = g.Galaxy(
    redshift=0.5,
    mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.9, phi=90.0),
    shear=mp.ExternalShear(magnitude=0.05, phi=90.0))

source_galaxy_0 = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(centre=(0.25, 0.15), axis_ratio=0.7, phi=120.0, intensity=0.7, effective_radius=0.7,
                              sersic_index=1.0))

source_galaxy_1 = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(centre=(0.7, -0.5), axis_ratio=0.9, phi=60.0, intensity=0.2, effective_radius=1.6,
                              sersic_index=3.0))

tracer = ray_tracing.TracerImageSourcePlanes(
    lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy_0, source_galaxy_1],
    image_plane_grid_stack=image_plane_grid_stack)

simulated_ccd = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
    image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=pixel_scale,
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

ccd.output_ccd_data_to_fits(
    ccd_data=simulated_ccd,
    image_path=data_path + 'image.fits',
    psf_path=data_path + 'psf.fits',
    noise_map_path=data_path + 'noise_map.fits',
    overwrite=True)

####################################################

data_type = 'example'
data_name = 'lens_bulge_disk_mass_and_x1_source'

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

lens_galaxy = g.Galaxy(
    redshift=0.5,
    bulge=lp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, intensity=0.3, effective_radius=0.6,
                              sersic_index=3.0),
    disk=lp.EllipticalExponential(centre=(0.0, 0.0), axis_ratio=0.7, phi=30.0, intensity=0.2, effective_radius=1.6),
    mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0),
    shear=mp.ExternalShear(magnitude=0.05, phi=90.0))

source_galaxy = g.Galaxy(
    redshift=1.0,
    light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0, intensity=0.3, effective_radius=1.0,
                              sersic_index=2.5))

tracer = ray_tracing.TracerImageSourcePlanes(
    lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy], image_plane_grid_stack=image_plane_grid_stack)

simulated_ccd = simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
    image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=pixel_scale,
    exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

# Lets plot the simulated CCD data before we output it to files.
ccd_plotters.plot_ccd_subplot(ccd_data=simulated_ccd)

ccd.output_ccd_data_to_fits(
    ccd_data=simulated_ccd,
    image_path=data_path + 'image.fits',
    psf_path=data_path + 'psf.fits',
    noise_map_path=data_path + 'noise_map.fits',
    overwrite=True)