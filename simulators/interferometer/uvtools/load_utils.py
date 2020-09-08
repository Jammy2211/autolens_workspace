from astropy import units
from astropy.io import fits
import os

from autolens_workspace.simulators.interferometer.uvtools import casa_utils

directory = os.path.dirname(os.path.realpath(__file__))


def uv_wavelengths_single_channel():

    central_frequency = 260 * units.GHz
    frequencies = central_frequency.to(units.Hz).value

    uv = fits.getdata(filename=f"{directory}/uv_single_12000s.fits")

    return casa_utils.convert_uv_coords_from_meters_to_wavelengths(
        uv=uv, frequencies=frequencies
    )


def uv_wavelengths_channel_averaging():

    central_frequency = 260 * units.GHz

    frequencies = casa_utils.generate_frequencies(
        central_frequency=central_frequency, n_channels=128
    )

    uv = fits.getdata(filename=f"{directory}/uv_120s.fits")

    return casa_utils.convert_uv_coords_from_meters_to_wavelengths(
        uv=uv, frequencies=frequencies
    )
