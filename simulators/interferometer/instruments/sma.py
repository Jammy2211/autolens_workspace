@classmethod
def sma(
    cls,
    real_space_shape_2d=(151, 151),
    real_space_pixel_scales=(0.05, 0.05),
    sub_size=8,
    primary_beam_shape_2d=None,
    primary_beam_sigma=None,
    exposure_time=100.0,
    background_level=1.0,
    noise_sigma=0.1,
    noise_if_add_noise_false=0.1,
    noise_seed=-1,
):
    """Default settings for an observation with the Large Synotpic Survey Telescope.

    This can be customized by over-riding the default input values."""

    uv_wavelengths_path = "{}/dataset/sma_uv_wavelengths.fits".format(
        os.path.dirname(os.path.realpath(__file__))
    )

    uv_wavelengths = array_util.numpy_array_1d_from_fits(
        file_path=uv_wavelengths_path, hdu=0
    )

    if primary_beam_shape_2d is not None and primary_beam_sigma is not None:
        primary_beam = kernel.Kernel.from_gaussian(
            shape_2d=primary_beam_shape_2d,
            sigma=primary_beam_sigma,
            pixel_scales=real_space_pixel_scales,
        )
    else:
        primary_beam = None

    return cls(
        real_space_shape_2d=real_space_shape_2d,
        real_space_pixel_scales=real_space_pixel_scales,
        uv_wavelengths=uv_wavelengths,
        sub_size=sub_size,
        primary_beam=primary_beam,
        exposure_time=exposure_time,
        background_level=background_level,
        noise_sigma=noise_sigma,
        noise_if_add_noise_false=noise_if_add_noise_false,
        noise_seed=noise_seed,
    )
