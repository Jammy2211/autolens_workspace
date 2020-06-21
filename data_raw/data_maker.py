import autofit as af
import autolens as al
import autolens.plot as aplt

import numpy as np
import os

# This script takes te file you sent me, which is in the path:

# 'autolens_workspace/data_raw/pisco/'

# and creates a dataset which we will fit in the pisco runner. The main purpose of this script is to set up the
# noise map and psf.

workspace_path = "{}/..".format(os.path.dirname(os.path.realpath(__file__)))
data_raw_path = af.util.create_path(path=workspace_path, folders=["data_raw"])

# The image is in very weird units (super high numbers). I don't know what they mean, but I made them lower to help
# autolens.

# Ideally you'll convert the image to electrons per second!

image_file = f"{data_raw_path}Blue_Eyebrow_g_cutout_141.fits"
image = al.Array.from_fits(file_path=image_file, pixel_scales=0.11)

# I have no idea what the correct noise map is, again I'm making my numbers up!
noise_map_file = f"{data_raw_path}Blue_Eyebrow_g_noise_141.fits"
noise_map = al.Array.from_fits(file_path=noise_map_file, pixel_scales=0.11)

# Anddddddddddd I also don't know the PSF :P
psf_file = f"{data_raw_path}Blue_Eyebrow_g_STAR2_norm.fits"
psf = al.Kernel.from_fits(file_path=psf_file, hdu=0, pixel_scales=0.11)

# This sets up the files above as an imaging object, which we will output for analysis with autolens.

imaging = al.Imaging(image=image, noise_map=noise_map, psf=psf)

dataset_name = "pisco_0"

# Now we're going to output this processed data to 'autolens_workspace/dataset/pisco/'.

pisco_output_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_name]
)

imaging.output_to_fits(
    image_path=pisco_output_path + "image.fits",
    psf_path=pisco_output_path + "psf.fits",
    noise_map_path=pisco_output_path + "noise_map.fits",
    overwrite=True,
)

pisco_image_path = af.util.create_path(path=pisco_output_path, folders=["images"])

aplt.Imaging.subplot_imaging(imaging=imaging)

aplt.Imaging.subplot_imaging(
    imaging=imaging,
    sub_plotter=aplt.SubPlotter(
        output=aplt.Output(path=pisco_image_path, format="png")
    ),
)

aplt.Imaging.individual(
    imaging=imaging,
    plot_image=True,
    plot_noise_map=True,
    plot_psf=True,
    plot_signal_to_noise_map=True,
    plot_absolute_signal_to_noise_map=True,
    plot_potential_chi_squared_map=True,
    plotter=aplt.Plotter(output=aplt.Output(path=pisco_image_path, format="png")),
)
