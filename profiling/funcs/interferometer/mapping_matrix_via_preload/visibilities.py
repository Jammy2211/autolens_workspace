import autolens as al

import numba

import time

import numpy as np

import pytest


@numba.jit(nopython=True, cache=False, parallel=True)
def real_transformed_mapping_matrix_via_preload_jit(mapping_matrix, preloaded_reals):

    transformed_vis = np.zeros(preloaded_reals.shape[1])

    transfomed_mapping_matrix = np.zeros(
        (preloaded_reals.shape[1], mapping_matrix.shape[1])
    )

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):

            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:

                for vis_1d_index in range(preloaded_reals.shape[1]):

                    transformed_vis[vis_1d_index] += preloaded_reals[
                        image_1d_index, vis_1d_index
                    ]

                # transfomed_mapping_matrix[vis_1d_index, pixel_1d_index] += (
                #     value * preloaded_reals[image_1d_index, vis_1d_index]
                # )

    return transfomed_mapping_matrix


@numba.jit(nopython=True, cache=False, parallel=True)
def real_transformed_mapping_matrix_via_preload_jit_cheat(
    mapping_matrix, preloaded_reals
):

    transfomed_mapping_matrix = np.zeros((preloaded_reals.shape[1]))

    for pixel_1d_index in range(mapping_matrix.shape[1]):
        for image_1d_index in range(mapping_matrix.shape[0]):

            value = mapping_matrix[image_1d_index, pixel_1d_index]

            if value > 0:

                for vis_1d_index in range(preloaded_reals.shape[1]):
                    transfomed_mapping_matrix[vis_1d_index] += (
                        value * preloaded_reals[0, vis_1d_index]
                    )

    return transfomed_mapping_matrix


print("Description: numba functions with preload, no prange and default functions.")

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

pixelization = al.pix.VoronoiMagnification(shape=(30, 30))

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])


repeats = 1

print("Number of repeats = ", repeats)

for total_visibilities in [
    100,
    1000,
    5000,
    10000,
    25000,
]:  # , 50000, 100000, 500000, 1000000, 2000000]:

    print()
    print("Number of visibilities = " + str(total_visibilities) + "\n")

    shape_data = 8 * total_visibilities
    shape_preloads = total_visibilities * image_pixels * 2

    total_shape = shape_data + shape_preloads

    visibilities = al.visibilities.ones(shape_1d=(total_visibilities,))
    uv_wavelengths = np.ones(shape=(total_visibilities, 2))

    interferometer = al.interferometer(
        visibilities=visibilities,
        noise_map=al.visibilities.ones(shape_1d=(total_visibilities,)),
        uv_wavelengths=uv_wavelengths,
    )

    masked_interferometer = al.masked.interferometer(
        interferometer=interferometer,
        real_space_mask=real_space_mask,
        visibilities_mask=np.full(fill_value=False, shape=(total_visibilities,)),
    )

    traced_grid = tracer.traced_grids_of_planes_from_grid(
        grid=masked_interferometer.grid
    )[-1]

    traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
        grid=masked_interferometer.grid
    )[-1]

    mapper = pixelization.mapper_from_grid_and_sparse_grid(
        grid=traced_grid, sparse_grid=traced_sparse_grid, inversion_uses_border=True
    )

    mapping_matrix = mapper.mapping_matrix

    if total_visibilities == 100:

        transformed_mapping_matrix = al.util.transformer.real_transformed_mapping_matrix_via_preload_jit(
            mapping_matrix=mapping_matrix,
            preloaded_reals=masked_interferometer.transformer.preload_real_transforms,
        )

        transformed_mapping_matrix_manual = real_transformed_mapping_matrix_via_preload_jit(
            mapping_matrix=mapping_matrix,
            preloaded_reals=masked_interferometer.transformer.preload_real_transforms,
        )

        # assert transformed_mapping_matrix == pytest.approx(
        #     transformed_mapping_matrix_manual, 1.0e-4
        # )

    start = time.time()
    for i in range(repeats):
        transformed_mapping_matrix_manual = real_transformed_mapping_matrix_via_preload_jit(
            mapping_matrix=mapping_matrix,
            preloaded_reals=masked_interferometer.transformer.preload_real_transforms,
        )
    diff = time.time() - start
    print(
        "Time to transformed compute mapping matrix = {}".format(2.0 * diff / repeats)
    )

    start = time.time()
    for i in range(repeats):
        np.sum(masked_interferometer.transformer.preload_real_transforms)
    diff = time.time() - start
    print("Time to sum preloaded transforms = {}".format(2.0 * diff / repeats))
