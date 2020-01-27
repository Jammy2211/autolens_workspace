import autofit as af
from astropy.io import fits
import os

import autolens as al
import autolens.plot as aplt


def simulate_imaging():

    psf = al.kernel.from_gaussian(shape_2d=(21, 21), sigma=0.05, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


def simulate_imaging_in_counts():

    psf = al.kernel.from_gaussian(shape_2d=(21, 21), sigma=0.05, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=1000 * 0.3,
            effective_radius=1.0,
            sersic_index=2.0,
        ),
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=1000 * 0.2,
            effective_radius=1.0,
            sersic_index=1.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


def simulate_imaging_with_large_stamp():

    psf = al.kernel.from_gaussian(shape_2d=(21, 21), sigma=0.05, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


def simulate_imaging_with_small_stamp():

    psf = al.kernel.from_gaussian(shape_2d=(21, 21), sigma=0.05, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


def simulate_imaging_with_offset_centre():

    psf = al.kernel.from_gaussian(shape_2d=(21, 21), sigma=0.05, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(1.0, 1.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mp.SphericalIsothermal(centre=(1.0, 1.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalSersic(
            centre=(1.0, 1.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


def simulate_imaging_with_large_psf():

    psf = al.kernel.from_gaussian(shape_2d=(101, 101), sigma=0.05, pixel_scales=0.1)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


def simulate_imaging_with_psf_with_offset_centre():

    psf = al.kernel.from_gaussian(
        shape_2d=(21, 21), sigma=0.05, pixel_scales=0.1, centre=(0.1, 0.1)
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.3, effective_radius=1.0, sersic_index=2.0
        ),
        mass=al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.SphericalSersic(
            centre=(0.0, 0.0), intensity=0.2, effective_radius=1.0, sersic_index=1.5
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.simulator.imaging(
        shape_2d=(130, 130),
        pixel_scales=0.1,
        exposure_time=300.0,
        sub_size=1,
        psf=psf,
        background_level=0.1,
        add_noise=True,
    )

    return simulator.from_tracer(tracer=tracer)


def simulate_all_imaging(dataset_path):

    imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=dataset_path, folder_names=["imaging"]
    )

    imaging = simulate_imaging()

    imaging.output_to_fits(
        image_path=imaging_path + "image.fits",
        noise_map_path=imaging_path + "noise_map.fits",
        psf_path=imaging_path + "psf.fits",
        overwrite=True,
    )

    new_hdul = fits.HDUList()
    new_hdul.append(fits.ImageHDU(imaging.image.in_2d))
    new_hdul.append(fits.ImageHDU(imaging.noise_map.in_2d))
    new_hdul.append(fits.ImageHDU(imaging.psf.in_2d))
    new_hdul.append(fits.ImageHDU(imaging.exposure_time_map.in_2d))

    if os.path.exists(imaging_path + "multiple_hdus.fits"):
        os.remove(imaging_path + "multiple_hdus.fits")

    new_hdul.writeto(imaging_path + "multiple_hdus.fits")

    imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=dataset_path, folder_names=["imaging_in_counts"]
    )

    imaging_in_counts = simulate_imaging_in_counts()

    imaging_in_counts.output_to_fits(
        image_path=imaging_path + "image.fits",
        noise_map_path=imaging_path + "noise_map.fits",
        psf_path=imaging_path + "psf.fits",
        exposure_time_map_path=imaging_path + "exposure_time_map.fits",
        overwrite=True,
    )

    imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=dataset_path, folder_names=["imaging_with_large_stamp"]
    )

    imaging_with_large_stamp = simulate_imaging_with_large_stamp()

    imaging_with_large_stamp.output_to_fits(
        image_path=imaging_path + "image.fits",
        noise_map_path=imaging_path + "noise_map.fits",
        psf_path=imaging_path + "psf.fits",
        overwrite=True,
    )

    imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=dataset_path, folder_names=["imaging_with_small_stamp"]
    )

    imaging_with_small_stamp = simulate_imaging_with_small_stamp()

    imaging_with_small_stamp.output_to_fits(
        image_path=imaging_path + "image.fits",
        noise_map_path=imaging_path + "noise_map.fits",
        psf_path=imaging_path + "psf.fits",
        overwrite=True,
    )

    imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=dataset_path, folder_names=["imaging_offset_centre"]
    )

    imaging_offset_centre = simulate_imaging_with_offset_centre()

    imaging_offset_centre.output_to_fits(
        image_path=imaging_path + "image.fits",
        noise_map_path=imaging_path + "noise_map.fits",
        psf_path=imaging_path + "psf.fits",
        overwrite=True,
    )

    imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=dataset_path, folder_names=["imaging_with_large_psf"]
    )

    imaging_with_large_psf = simulate_imaging_with_large_psf()

    imaging_with_large_psf.output_to_fits(
        image_path=imaging_path + "image.fits",
        noise_map_path=imaging_path + "noise_map.fits",
        psf_path=imaging_path + "psf.fits",
        overwrite=True,
    )

    imaging_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=dataset_path, folder_names=["imaging_with_off_centre_psf"]
    )

    imaging_with_off_centre_psf = simulate_imaging_with_psf_with_offset_centre()

    imaging_with_off_centre_psf.output_to_fits(
        image_path=imaging_path + "image.fits",
        noise_map_path=imaging_path + "noise_map.fits",
        psf_path=imaging_path + "psf.fits",
        overwrite=True,
    )
