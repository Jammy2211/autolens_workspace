import autolens as al

import numba

import time

import numpy as np

import pytest


@numba.jit(nopython=True, cache=False, parallel=True)
def flip_transformed_mapping_matrix(transformed_mapping_matrix):

    flipped = np.zeros(
        shape=(transformed_mapping_matrix.shape[1], transformed_mapping_matrix.shape[0])
    )

    for i in range(transformed_mapping_matrix.shape[1]):
        for j in range(transformed_mapping_matrix.shape[0]):
            flipped[i, j] = transformed_mapping_matrix[j, i]

    return flipped


@numba.jit(nopython=True, cache=False, parallel=True)
def curvature_matrix_from_transformed_mapping_matrix(
    transformed_mapping_matrix, noise_map
):
    """Compute the curvature matrix *F* from a transformed util matrix *f* and the 1D noise-map *\sigma* \
    (see Warren & Dye 2003).

    Parameters
    -----------
    transformed_mapping_matrix : ndarray
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    noise_map : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    flist : ndarray
        NumPy array of floats used to store mappings for efficienctly calculation.
    iflist : ndarray
        NumPy array of integers used to store mappings for efficienctly calculation.
    """
    curvature_matrix = np.zeros(
        (transformed_mapping_matrix.shape[1], transformed_mapping_matrix.shape[1])
    )

    for pix_1d_index_0 in range(transformed_mapping_matrix.shape[1]):
        for pix_1d_index_1 in range(pix_1d_index_0 + 1):
            for vis_1d_index in range(transformed_mapping_matrix.shape[0]):
                curvature_matrix[pix_1d_index_0, pix_1d_index_1] += (
                    transformed_mapping_matrix[vis_1d_index, pix_1d_index_0]
                    * transformed_mapping_matrix[vis_1d_index, pix_1d_index_1]
                    / noise_map[vis_1d_index] ** 2
                )

    for i in range(transformed_mapping_matrix.shape[1]):
        for j in range(transformed_mapping_matrix.shape[1]):
            curvature_matrix[i, j] = curvature_matrix[j, i]

    return curvature_matrix


@numba.jit(nopython=True, cache=False, parallel=True)
def curvature_matrix_from_transformed_mapping_matrix_edit(
    transformed_mapping_matrix, noise_map_inv_sq
):
    """Compute the curvature matrix *F* from a transformed util matrix *f* and the 1D noise-map *\sigma* \
    (see Warren & Dye 2003).

    Parameters
    -----------
    transformed_mapping_matrix : ndarray
        The matrix representing the transformed mappings between sub-grid pixels and pixelization pixels.
    noise_map_inv_sq : ndarray
        Flattened 1D array of the noise-map used by the inversion during the fit.
    flist : ndarray
        NumPy array of floats used to store mappings for efficienctly calculation.
    iflist : ndarray
        NumPy array of integers used to store mappings for efficienctly calculation.
    """
    curvature_matrix = np.zeros(
        (transformed_mapping_matrix.shape[0], transformed_mapping_matrix.shape[0])
    )

    for pix_1d_index_0 in numba.prange(transformed_mapping_matrix.shape[0]):
        for pix_1d_index_1 in range(pix_1d_index_0 + 1):

            curvature_matrix[pix_1d_index_0, pix_1d_index_1] = np.sum(
                transformed_mapping_matrix[pix_1d_index_0, :]
                * transformed_mapping_matrix[pix_1d_index_1, :]
                * noise_map_inv_sq[:] ** 2
            )

    for i in range(transformed_mapping_matrix.shape[0]):
        for j in range(transformed_mapping_matrix.shape[0]):
            curvature_matrix[i, j] = curvature_matrix[j, i]

    return curvature_matrix


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
]:  # , 50000, 100000]:#, 500000, 1000000, 2000000]:

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

    start_overall = time.time()

    mapping_matrix = mapper.mapping_matrix

    transformed_mapping_matrices = masked_interferometer.transformer.transformed_mapping_matrices_from_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    transformed_mapping_matrices_flipped = [
        flip_transformed_mapping_matrix(transformed_mapping_matrix=matrix)
        for matrix in transformed_mapping_matrices
    ]

    noise_map_inv_sq = (1.0 / masked_interferometer.noise_map) ** 2.0

    if total_visibilities == 100:
        real_curvature_matrix = curvature_matrix_from_transformed_mapping_matrix(
            transformed_mapping_matrix=transformed_mapping_matrices[0],
            noise_map=masked_interferometer.noise_map[:, 0],
        )
        real_curvature_matrix_edit = curvature_matrix_from_transformed_mapping_matrix_edit(
            transformed_mapping_matrix=transformed_mapping_matrices_flipped[0],
            noise_map_inv_sq=noise_map_inv_sq[:, 0],
        )
        assert real_curvature_matrix == pytest.approx(
            real_curvature_matrix_edit, 1.0e-4
        )

    start = time.time()
    for i in range(repeats):
        real_curvature_matrix = curvature_matrix_from_transformed_mapping_matrix_edit(
            transformed_mapping_matrix=transformed_mapping_matrices_flipped[0],
            noise_map_inv_sq=noise_map_inv_sq[:, 0],
        )
    diff = time.time() - start
    print("Time to compute curvature matrix = {}".format(2.0 * diff / repeats))
