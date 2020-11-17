from astropy.io import fits
from os import path
import uv_util
import numpy as np
from astropy import constants, units as au
import autolens as al


def convert_array_to_wavelengths(array, frequency):

    array_converted = ((array * au.m) * (frequency * au.Hz) / constants.c).decompose()

    return array_converted.value


def convert_uv_coords_from_meters_to_wavelengths(uv, frequencies):

    if np.shape(frequencies):

        u_wavelengths, v_wavelengths = np.zeros(
            shape=(2, len(frequencies), uv.shape[1])
        )

        for i in range(len(frequencies)):
            u_wavelengths[i, :] = convert_array_to_wavelengths(
                array=uv[0, :], frequency=frequencies[i]
            )
            v_wavelengths[i, :] = convert_array_to_wavelengths(
                array=uv[1, :], frequency=frequencies[i]
            )

    else:

        u_wavelengths = convert_array_to_wavelengths(
            array=uv[0, :], frequency=frequencies
        )
        v_wavelengths = convert_array_to_wavelengths(
            array=uv[1, :], frequency=frequencies
        )

    return np.stack(arrays=(u_wavelengths, v_wavelengths), axis=-1)


"""Load the ALMA UV baselines."""

uv_full = fits.getdata(
    filename=path.join("simulators", "interferometer", "uv_wavelengths", "alma_uv.fits")
)

uv_full = np.transpose(
    convert_uv_coords_from_meters_to_wavelengths(
        uv=uv_full, frequencies=260 * au.GHz.to(au.Hz)
    )
)


"""
These settings control the total integration time, time per visibility observations and number of
visibilities observed in each channel.
"""

total_integration_time = 10800
time_per_visibility = 1
total_visibilities_per_channel = int(total_integration_time / time_per_visibility)

"""
Variables to trim the number of visibilities observed and thus observe datasets with different 
total number of visibilities.
"""

time_trim_min = 0
time_trim_max = int(total_integration_time / 1.0)
uv_util.check_time_steps(
    t_int=time_per_visibility, t_trim_max=time_trim_max, t_trim_min=time_trim_min
)

j_trim_min = int(time_trim_min / time_per_visibility)
j_trim_max = int(time_trim_max / time_per_visibility)

uv_reshaped = uv_full.reshape(
    (
        2,
        total_visibilities_per_channel,
        int(uv_full.shape[-1] / total_visibilities_per_channel),
    )
)

uv_reshaped_trimmed = uv_reshaped[:, j_trim_min:j_trim_max, :]

uv_wavelengths = uv_reshaped_trimmed.reshape(
    (
        int(uv_reshaped_trimmed.shape[1] * uv_reshaped_trimmed.shape[2]),
        uv_reshaped_trimmed.shape[0],
    )
)

print(uv_wavelengths.shape)

al.util.array.numpy_array_2d_to_fits(
    array_2d=uv_wavelengths,
    file_path=path.join(
        "simulators",
        "interferometer",
        "uv_wavelengths",
        f"alma_uv_wavelengths_x{uv_wavelengths.shape[0]}.fits",
    ),
)
