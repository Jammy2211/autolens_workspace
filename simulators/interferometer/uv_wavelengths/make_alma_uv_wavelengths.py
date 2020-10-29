from astropy.io import fits
import numpy as np

import autolens as al
from autolens_workspace.simulators.interferometer.uv_wavelengths import uv_util

"""Load the ALMA UV baselines."""

uv_full = fits.getdata(
    filename=f"simulators/interferometer/uv_wavelengths/alma_uv.fits"
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
        uv_reshaped_trimmed.shape[0],
        int(uv_reshaped_trimmed.shape[1] * uv_reshaped_trimmed.shape[2]),
    )
)

print(uv_wavelengths.shape)

al.util.array.numpy_array_2d_to_fits(
    array_2d=uv_wavelengths,
    file_path=f"simulators/interferometer/uv_wavelengths/"
    f"alma_uv_wavelengths_x{uv_wavelengths.shape[1]}.fits",
)
