from astropy import units
from astropy.io import fits
import os

from autolens_workspace.simulators.interferometer.uvtools import casa_utils

directory = os.path.dirname(os.path.realpath(__file__))


def uv_wavelengths_single_channel():

    central_frequency = 260 * units.GHz
    frequencies = central_frequency.to(units.Hz).value

    uv = fits.getdata(filename=f"{directory}/alma_uv_wavelengths.fits")

    return casa_utils.convert_uv_coords_from_meters_to_wavelengths(
        uv=uv, frequencies=frequencies
    )
