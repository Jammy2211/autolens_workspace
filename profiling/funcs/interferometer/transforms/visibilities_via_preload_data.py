import autofit as af
import autolens as al

import os
import numba

import time

import numpy as np


@numba.jit(nopython=True, cache=False, parallel=True)
def preload_real_transforms(grid_radians, uv_wavelengths):

    preloaded_real_transforms = np.zeros(
        shape=(grid_radians.shape[0], uv_wavelengths.shape[0])
    )

    for image_1d_index in range(grid_radians.shape[0]):
        for vis_1d_index in range(uv_wavelengths.shape[0]):
            preloaded_real_transforms[image_1d_index, vis_1d_index] += np.cos(
                -2.0
                * np.pi
                * (
                    grid_radians[image_1d_index, 1] * uv_wavelengths[vis_1d_index, 0]
                    + grid_radians[image_1d_index, 0] * uv_wavelengths[vis_1d_index, 1]
                )
            )

    return preloaded_real_transforms


@numba.jit(nopython=True, cache=False, parallel=True)
def real_visibilities_from_image_via_preload(image_1d, preloaded_reals):

    real_visibilities = np.zeros(shape=(preloaded_reals.shape[1]))

    for image_1d_index in range(image_1d.shape[0]):
        for vis_1d_index in range(preloaded_reals.shape[1]):
            real_visibilities[vis_1d_index] += (
                image_1d[image_1d_index] * preloaded_reals[image_1d_index, vis_1d_index]
            )

    return real_visibilities


print("Description: numab functions with preload, no prange and default functions.")

real_space_shape_2d = (151, 151)
real_space_pixel_scales = 0.05
real_space_sub_size = 1
real_space_radius = 3.0

real_space_mask = al.mask.circular(
    shape_2d=real_space_shape_2d,
    pixel_scales=real_space_pixel_scales,
    radius=real_space_radius,
)

real_space_grid = al.grid.from_mask(mask=real_space_mask)
real_space_grid_radians = real_space_grid.in_radians.in_1d_binned

print("Real space sub grid size = " + str(real_space_sub_size))
print("Real space circular mask radius = " + str(real_space_radius) + "\n")
print("Number of points = " + str(real_space_grid.sub_shape_1d) + "\n")

image_pixels = real_space_shape_2d[0] * real_space_shape_2d[1]

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.8,
        phi=60.0,
        intensity=0.4,
        effective_radius=0.5,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

repeats = 3

print("Number of repeats = ", repeats)

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

dataset_label = "arisuv"
dataset_size = "mil_05"

dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label]
)

interferometer = al.interferometer.from_fits(
    visibilities_path=dataset_path + "visibilities.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    uv_wavelengths_path=dataset_path + "uv_wavelengths.fits",
)

total_visibilities = interferometer.visibilities.shape[0]

for total_visibilities in [
    100,
    1000,
    5000,
    10000,
    25000,
]:  # , 50000, 100000, 500000, 1000000, 2000000]:

    print()
    print("########################")
    print()
    print("Number of visibilities = " + str(total_visibilities) + "\n")

    shape_data = 8 * total_visibilities
    shape_preloads = total_visibilities * image_pixels * 2

    total_shape = shape_data + shape_preloads

    print("Data Memory Use (GB) = " + str(shape_data * 8e-9))
    print("PreLoad Memory Use (GB) = " + str(shape_preloads * 8e-9))
    print("Total Memory Use (GB) = " + str(total_shape * 8e-9))
    print()

    visibilities = al.visibilities.manual_1d(
        visibilities=interferometer.visibilities[:total_visibilities, :]
    )
    uv_wavelengths = interferometer.uv_wavelengths[:total_visibilities, :]

    start_overall = time.time()

    start = time.time()
    preloaded_real_transforms = preload_real_transforms(
        grid_radians=real_space_grid_radians, uv_wavelengths=uv_wavelengths
    )

    diff = time.time() - start
    print("Time to PreLoad Transforms (1_iteration) = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
        profile_image = tracer.profile_image_from_grid(grid=real_space_grid)
    diff = time.time() - start
    print("Time to create profile image = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        real_visibilities_from_image_via_preload(
            image_1d=profile_image, preloaded_reals=preloaded_real_transforms
        )
    diff = time.time() - start
    print("Time to perform fourier transform = {}".format(2.0 * diff / repeats))
