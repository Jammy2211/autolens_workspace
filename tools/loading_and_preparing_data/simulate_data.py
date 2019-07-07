import autofit as af
from astropy.io import fits
import os

from autolens.data import ccd
from autolens.data import simulated_ccd
from autolens.model.profiles import light_profiles as lp, mass_profiles as mp


def simulate_ccd_data():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(
        shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
        shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0),
        mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1,
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_ccd_data_in_counts():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(
        shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
        shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=1000*0.3, effective_radius=1.0, sersic_index=2.0),
        mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=1000*0.2, effective_radius=1.0, sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1,
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_ccd_data_with_large_stamp():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(
        shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
        shape=(500, 500), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0),
        mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1,
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_ccd_data_with_small_stamp():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(
        shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
        shape=(50, 50), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0),
        mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1,
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_ccd_data_with_offset_centre():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(
        shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
        shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(1.0, 1.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0),
        mass=mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(1.0, 1.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1,
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_ccd_data_with_large_psf():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(
        shape=(101, 101), sigma=0.05, pixel_scale=0.1)

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
        shape=(100, 100), pixel_scale=0.1, psf_shape=(101, 101))

    lens_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0),
        mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1,
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)

def simulate_ccd_data_with_psf_with_offset_centre():

    from autolens.data.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = ccd.PSF.from_gaussian(
        shape=(21, 21), sigma=0.05, pixel_scale=0.1, centre=(0.1, 0.1))

    image_plane_grid_stack = grids.GridStack.grid_stack_for_simulation(
        shape=(100, 100), pixel_scale=0.1, psf_shape=(21, 21))

    lens_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0),
        mass=mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2))

    source_galaxy = g.Galaxy(
        light=lp.SphericalSersic(centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5))

    tracer = ray_tracing.TracerImageSourcePlanes(
        lens_galaxies=[lens_galaxy], source_galaxies=[source_galaxy],
        image_plane_grid_stack=image_plane_grid_stack)

    return simulated_ccd.SimulatedCCDData.from_image_and_exposure_arrays(
        image=tracer.profile_image_plane_image_2d_for_simulation, pixel_scale=0.1,
        exposure_time=300.0, psf=psf, background_sky_level=0.1, add_noise=True)


def simulate_all_ccd_data(data_path):
    
    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=['ccd_data'])

    ccd_data = simulate_ccd_data()

    ccd.output_ccd_data_to_fits(
        ccd_data=ccd_data,
        image_path=ccd_data_path + 'image.fits',
        noise_map_path=ccd_data_path + 'noise_map.fits',
        psf_path=ccd_data_path + 'psf.fits',
        overwrite=True)

    new_hdul = fits.HDUList()
    new_hdul.append(fits.ImageHDU(ccd_data.image_2d))
    new_hdul.append(fits.ImageHDU(ccd_data.noise_map))
    new_hdul.append(fits.ImageHDU(ccd_data.psf))
    new_hdul.append(fits.ImageHDU(ccd_data.exposure_time_map))

    if os.path.exists(ccd_data_path + 'multiple_hdus.fits'):
        os.remove(ccd_data_path + 'multiple_hdus.fits')

    new_hdul.writeto(ccd_data_path + 'multiple_hdus.fits')

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=['ccd_data_in_counts'])

    ccd_data_in_counts = simulate_ccd_data_in_counts()

    ccd.output_ccd_data_to_fits(
        ccd_data=ccd_data_in_counts,
        image_path=ccd_data_path + 'image.fits',
        noise_map_path=ccd_data_path + 'noise_map.fits',
        psf_path=ccd_data_path + 'psf.fits',
        exposure_time_map_path=ccd_data_path + 'exposure_time_map.fits',
        overwrite=True)

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=['ccd_data_with_large_stamp'])

    ccd_data_with_large_stamp = simulate_ccd_data_with_large_stamp()

    ccd.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_large_stamp,
        image_path=ccd_data_path + 'image.fits',
        noise_map_path=ccd_data_path + 'noise_map.fits',
        psf_path=ccd_data_path + 'psf.fits',
        overwrite=True)

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=['ccd_data_with_small_stamp'])

    ccd_data_with_small_stamp = simulate_ccd_data_with_small_stamp()

    ccd.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_small_stamp,
        image_path=ccd_data_path + 'image.fits',
        noise_map_path=ccd_data_path + 'noise_map.fits',
        psf_path=ccd_data_path + 'psf.fits',
        overwrite=True)

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=['ccd_data_offset_centre'])

    ccd_data_offset_centre = simulate_ccd_data_with_offset_centre()

    ccd.output_ccd_data_to_fits(
        ccd_data=ccd_data_offset_centre,
        image_path=ccd_data_path + 'image.fits',
        noise_map_path=ccd_data_path + 'noise_map.fits',
        psf_path=ccd_data_path + 'psf.fits',
        overwrite=True)

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=['ccd_data_with_large_psf'])

    ccd_data_with_large_psf = simulate_ccd_data_with_large_psf()

    ccd.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_large_psf,
        image_path=ccd_data_path + 'image.fits',
        noise_map_path=ccd_data_path + 'noise_map.fits',
        psf_path=ccd_data_path + 'psf.fits',
        overwrite=True)

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=['ccd_data_with_off_centre_psf'])

    ccd_data_with_off_centre_psf = simulate_ccd_data_with_psf_with_offset_centre()

    ccd.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_off_centre_psf,
        image_path=ccd_data_path + 'image.fits',
        noise_map_path=ccd_data_path + 'noise_map.fits',
        psf_path=ccd_data_path + 'psf.fits',
        overwrite=True)