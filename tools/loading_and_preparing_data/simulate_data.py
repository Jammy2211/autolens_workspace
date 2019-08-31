import autofit as af
from astropy.io import fits
import os

import autolens as al


def simulate_ccd_data():

    psf = al.PSF.from_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(100, 100), pixel_scale=0.1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.2
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


def simulate_ccd_data_in_counts():

    psf = al.PSF.from_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(100, 100), pixel_scale=0.1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=1000 * 0.3,
            effective_radius=1.0,
            sersic_index=2.0,
        ),
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.2
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=1000 * 0.2,
            effective_radius=1.0,
            sersic_index=1.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


def simulate_ccd_data_with_large_stamp():

    from autolens.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = al.PSF.from_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(500, 500), pixel_scale=0.1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.2
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


def simulate_ccd_data_with_small_stamp():

    from autolens.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = al.PSF.from_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(50, 50), pixel_scale=0.1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.2
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


def simulate_ccd_data_with_offset_centre():

    from autolens.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = al.PSF.from_gaussian(shape=(21, 21), sigma=0.05, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(100, 100), pixel_scale=0.1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalSersic(
            centre=(1.0, 1.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.2
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(
            centre=(1.0, 1.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


def simulate_ccd_data_with_large_psf():

    from autolens.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = al.PSF.from_gaussian(shape=(101, 101), sigma=0.05, pixel_scale=0.1)

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(100, 100), pixel_scale=0.1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.2
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


def simulate_ccd_data_with_psf_with_offset_centre():

    from autolens.array import grids
    from autolens.model.galaxy import galaxy as g
    from autolens.lens import ray_tracing

    psf = al.PSF.from_gaussian(
        shape=(21, 21), sigma=0.05, pixel_scale=0.1, centre=(0.1, 0.1)
    )

    grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
        shape=(100, 100), pixel_scale=0.1
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mass_profiles.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.2
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.light_profiles.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    return al.SimulatedCCDData.from_tracer_grid_and_exposure_arrays(
        tracer=tracer,
        grid=grid,
        pixel_scale=0.1,
        exposure_time=300.0,
        psf=psf,
        background_sky_level=0.1,
        add_noise=True,
    )


def simulate_all_ccd_data(data_path):

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=["ccd_data"]
    )

    ccd_data = simulate_ccd_data()

    al.output_ccd_data_to_fits(
        ccd_data=ccd_data,
        image_path=ccd_data_path + "image.fits",
        noise_map_path=ccd_data_path + "noise_map.fits",
        psf_path=ccd_data_path + "psf.fits",
        overwrite=True,
    )

    new_hdul = fits.HDUList()
    new_hdul.append(fits.ImageHDU(ccd_data.image))
    new_hdul.append(fits.ImageHDU(ccd_data.noise_map))
    new_hdul.append(fits.ImageHDU(ccd_data.psf))
    new_hdul.append(fits.ImageHDU(ccd_data.exposure_time_map))

    if os.path.exists(ccd_data_path + "multiple_hdus.fits"):
        os.remove(ccd_data_path + "multiple_hdus.fits")

    new_hdul.writeto(ccd_data_path + "multiple_hdus.fits")

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=["ccd_data_in_counts"]
    )

    ccd_data_in_counts = simulate_ccd_data_in_counts()

    al.output_ccd_data_to_fits(
        ccd_data=ccd_data_in_counts,
        image_path=ccd_data_path + "image.fits",
        noise_map_path=ccd_data_path + "noise_map.fits",
        psf_path=ccd_data_path + "psf.fits",
        exposure_time_map_path=ccd_data_path + "exposure_time_map.fits",
        overwrite=True,
    )

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=["ccd_data_with_large_stamp"]
    )

    ccd_data_with_large_stamp = simulate_ccd_data_with_large_stamp()

    al.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_large_stamp,
        image_path=ccd_data_path + "image.fits",
        noise_map_path=ccd_data_path + "noise_map.fits",
        psf_path=ccd_data_path + "psf.fits",
        overwrite=True,
    )

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=["ccd_data_with_small_stamp"]
    )

    ccd_data_with_small_stamp = simulate_ccd_data_with_small_stamp()

    al.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_small_stamp,
        image_path=ccd_data_path + "image.fits",
        noise_map_path=ccd_data_path + "noise_map.fits",
        psf_path=ccd_data_path + "psf.fits",
        overwrite=True,
    )

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=["ccd_data_offset_centre"]
    )

    ccd_data_offset_centre = simulate_ccd_data_with_offset_centre()

    al.output_ccd_data_to_fits(
        ccd_data=ccd_data_offset_centre,
        image_path=ccd_data_path + "image.fits",
        noise_map_path=ccd_data_path + "noise_map.fits",
        psf_path=ccd_data_path + "psf.fits",
        overwrite=True,
    )

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=["ccd_data_with_large_psf"]
    )

    ccd_data_with_large_psf = simulate_ccd_data_with_large_psf()

    al.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_large_psf,
        image_path=ccd_data_path + "image.fits",
        noise_map_path=ccd_data_path + "noise_map.fits",
        psf_path=ccd_data_path + "psf.fits",
        overwrite=True,
    )

    ccd_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=data_path, folder_names=["ccd_data_with_off_centre_psf"]
    )

    ccd_data_with_off_centre_psf = simulate_ccd_data_with_psf_with_offset_centre()

    al.output_ccd_data_to_fits(
        ccd_data=ccd_data_with_off_centre_psf,
        image_path=ccd_data_path + "image.fits",
        noise_map_path=ccd_data_path + "noise_map.fits",
        psf_path=ccd_data_path + "psf.fits",
        overwrite=True,
    )
