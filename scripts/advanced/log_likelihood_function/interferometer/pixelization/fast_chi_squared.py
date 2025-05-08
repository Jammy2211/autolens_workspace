"""
__Log Likelihood Function: Fast Chi Squared__

This script describes how the `chi_squared` of an interferometer pixelization can be computed without using a
`transformed_mapping_matrix` or an NUFFT algorithm at all.

This means the likelihood function can be computed without ever performing an NUFFT, which for datasets of 10^6
visibilities or more can be extremely computationally expensive.

This can make the likelihood function significantly faster, for example with speed ups of hundreds of times or more
for tens or millions of visibilities. In fact, the run time does not scale with the number of visibilities at all,
meaning datasets of any size can be fitted in seconds.

It directly follows on from the `pixelization/log_likelihood_function.py` and ``pixelization/w_tilde.py` notebooks and
you should read through those examples before reading this script.

__Prerequisites__

You must read through the following likelihood functions first:

 - `pixelization/log_likelihood_function.py` the likelihood function for a pixelization.
 - `pixelization/w_tilde.py` the w-tilde formalism used to compute the likelihood function without an NUFFT.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
import numpy as np
from os import path

import autolens as al
import autolens.plot as aplt


"""
__Dataset__

Following the `pixelization/log_likelihood_function.py` script, we load and mask an `Imaging` dataset and
set oversampling to 1.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(80, 80), pixel_scales=0.05, radius=4.0
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=al.TransformerDFT,
)


"""
__W Tilde__

The fast chi-squared method uses the w-tilde matrix, which we compute now.
"""
from autoarray import numba_util


@numba_util.jit()
def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


w_tilde = w_tilde_curvature_interferometer_from(
    noise_map_real=np.array(dataset.noise_map.real),
    uv_wavelengths=np.array(dataset.uv_wavelengths),
    grid_radians_slim=np.array(dataset.grid.in_radians),
)


"""
__Mapping Matrix__

It also uses the `mapping_matrix` which we compute now.
"""
pixelization = al.Pixelization(
    mesh=al.mesh.Rectangular(shape=(30, 30)),
    regularization=al.reg.Constant(coefficient=1.0),
)

galaxy = al.Galaxy(redshift=0.5, pixelization=pixelization)

grid_rectangular = al.Mesh2DRectangular.overlay_grid(
    shape_native=galaxy.pixelization.mesh.shape, grid=dataset.grids.pixelization
)

mapper_grids = al.MapperGrids(
    mask=real_space_mask,
    source_plane_data_grid=dataset.grids.pixelization,
    source_plane_mesh_grid=grid_rectangular,
)

mapper = al.Mapper(
    mapper_grids=mapper_grids,
    regularization=None,
)

mapping_matrix = al.util.mapper.mapping_matrix_from(
    pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
    pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,  # unused for rectangular
    pix_weights_for_sub_slim_index=mapper.pix_weights_for_sub_slim_index,  # unused for rectangular
    pixels=mapper.pixels,
    total_mask_pixels=mapper.source_plane_data_grid.mask.pixels_in_mask,
    slim_index_for_sub_slim_index=mapper.slim_index_for_sub_slim_index,
    sub_fraction=np.array(mapper.over_sampler.sub_fraction),
)
