import numpy as np

from astropy import constants, units as au


# NOTE: THIS ONLY WORKS WHEN THE NUMBER OF CHANNELS IS EVEN NUMBER.
def generate_frequencies(central_frequency, n_channels, bandwidth=2.0 * au.GHz):

    df = (bandwidth / n_channels).to(au.Hz).value

    frequencies = np.arange(
        central_frequency.to(au.Hz).value - int(n_channels / 2.0) * df,
        central_frequency.to(au.Hz).value + int(n_channels / 2.0) * df,
        df,
    )

    return frequencies


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
