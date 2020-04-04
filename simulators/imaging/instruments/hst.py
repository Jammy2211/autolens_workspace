@classmethod
def hst(
    cls,
    pixel_scales=0.05,
    sub_size=8,
    psf_shape_2d=(31, 31),
    psf_sigma=0.05,
    exposure_time=2000.0,
    background_level=1.0,
    add_noise=True,
    noise_if_add_noise_false=0.1,
    noise_seed=-1,
):
    """Default settings for an observation with the Hubble Space Telescope.

    This can be customized by over-riding the default input values."""
    psf = kernel.Kernel.from_gaussian(
        shape_2d=psf_shape_2d, sigma=psf_sigma, pixel_scales=pixel_scales
    )
    return cls(
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        psf=psf,
        exposure_time=exposure_time,
        background_level=background_level,
        add_noise=add_noise,
        noise_if_add_noise_false=noise_if_add_noise_false,
        noise_seed=noise_seed,
    )
