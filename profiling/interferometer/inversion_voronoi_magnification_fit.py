import autolens as al

import numpy as np
import time

repeats = 1

total_visibilities = 10000
pixelization_shape_2d = (30, 30)

real_space_shape_2d = (100, 100)
real_space_pixel_scales = 0.05
real_space_sub_size = 1
real_space_radius = 3.0

image_pixels = real_space_shape_2d[0] * real_space_shape_2d[1]
source_pixels = pixelization_shape_2d[0] * pixelization_shape_2d[1]

shape_data = 8 * total_visibilities
shape_preloads = total_visibilities * image_pixels * 2
shape_mapping_matrix = total_visibilities * source_pixels

total_shape = shape_data + shape_preloads + shape_mapping_matrix

print("Data Memory Use (GB) = " + str(shape_data * 8e-9))
print("PreLoad Memory Use (GB) = " + str(shape_preloads * 8e-9))
print("Mapping Matrix Memory Use (GB) = " + str(shape_mapping_matrix * 8e-9))
print("Total Memory Use (GB) = " + str(total_shape * 8e-9))
print()

# Only delete this if the memory use looks... Okay
# stop

visibilities = al.visibilities.ones(shape_1d=(total_visibilities,))
uv_wavelengths = np.ones(shape=(total_visibilities, 2))
noise_map = al.visibilities.ones(shape_1d=(total_visibilities,))

interferometer = al.interferometer(
    visibilities=visibilities, noise_map=noise_map, uv_wavelengths=uv_wavelengths
)

print("Real space sub grid size = " + str(real_space_sub_size))
print("Real space circular mask radius = " + str(real_space_radius) + "\n")
print("pixelization shape = " + str(pixelization_shape_2d) + "\n")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0
    ),
)

pixelization = al.pix.VoronoiMagnification(shape=pixelization_shape_2d)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=pixelization,
    regularization=al.reg.Constant(coefficient=1.0),
)

mask = al.mask.circular(
    shape_2d=real_space_shape_2d,
    pixel_scales=real_space_pixel_scales,
    sub_size=real_space_sub_size,
    radius=real_space_radius,
)

masked_interferometer = al.masked.interferometer(
    interferometer=interferometer, real_space_mask=mask
)

print("Number of points = " + str(masked_interferometer.grid.sub_shape_1d) + "\n")
print(
    "Number of visibilities = "
    + str(masked_interferometer.visibilities.shape_1d)
    + "\n"
)

start_overall = time.time()

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

start = time.time()
for i in range(repeats):
    traced_grid = tracer.traced_grids_of_planes_from_grid(
        grid=masked_interferometer.grid
    )[-1]
diff = time.time() - start
print("Time to Setup Traced Grid = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    traced_sparse_grid = tracer.traced_sparse_grids_of_planes_from_grid(
        grid=masked_interferometer.grid
    )[-1]

diff = time.time() - start
print("Time to Setup Traced Sparse Grid = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    mapper = pixelization.mapper_from_grid_and_sparse_grid(
        grid=traced_grid, sparse_grid=traced_sparse_grid, inversion_uses_border=True
    )
diff = time.time() - start
print("Time to create mapper (Border Relocation) = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    mapping_matrix = mapper.mapping_matrix
diff = time.time() - start
print("Time to compute mapping matrix = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    transformed_mapping_matrices = masked_interferometer.transformer.transformed_mapping_matrices_from_mapping_matrix(
        mapping_matrix=mapping_matrix
    )
diff = time.time() - start
print("Time to compute transformed mapping matrices = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    real_data_vector = al.util.inversion.data_vector_from_transformed_mapping_matrix_and_data(
        transformed_mapping_matrix=transformed_mapping_matrices[0],
        visibilities=masked_interferometer.visibilities[:, 0],
        noise_map=masked_interferometer.noise_map[:, 0],
    )
diff = time.time() - start
print("Time to compute real data vector = {}".format(diff / repeats))


start = time.time()
for i in range(repeats):
    imag_data_vector = al.util.inversion.data_vector_from_transformed_mapping_matrix_and_data(
        transformed_mapping_matrix=transformed_mapping_matrices[1],
        visibilities=masked_interferometer.visibilities[:, 1],
        noise_map=masked_interferometer.noise_map[:, 1],
    )
diff = time.time() - start
print("Time to compute imaginary data vector = {}".format(diff / repeats))


start = time.time()
for i in range(repeats):
    real_curvature_matrix = al.util.inversion.curvature_matrix_from_transformed_mapping_matrix(
        transformed_mapping_matrix=transformed_mapping_matrices[0],
        noise_map=noise_map[:, 0],
    )
diff = time.time() - start
print("Time to compute real curvature matrix = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    imag_curvature_matrix = al.util.inversion.curvature_matrix_from_transformed_mapping_matrix(
        transformed_mapping_matrix=transformed_mapping_matrices[1],
        noise_map=noise_map[:, 1],
    )
diff = time.time() - start
print("Time to compute imaginary curvature matrix = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    regularization_matrix = al.util.regularization.constant_regularization_matrix_from_pixel_neighbors(
        coefficient=1.0,
        pixel_neighbors=mapper.pixelization_grid.pixel_neighbors,
        pixel_neighbors_size=mapper.pixelization_grid.pixel_neighbors_size,
    )
diff = time.time() - start
print("Time to compute reguarization matrix = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    real_curvature_reg_matrix = np.add(real_curvature_matrix, regularization_matrix)
    imag_curvature_reg_matrix = np.add(imag_curvature_matrix, regularization_matrix)
    data_vector = np.add(real_data_vector, imag_data_vector)
    curvature_reg_matrix = np.add(real_curvature_reg_matrix, imag_curvature_reg_matrix)
diff = time.time() - start
print("Time to compute curvature reguarization Matrices = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    reconstruction = np.linalg.solve(curvature_reg_matrix, data_vector)
diff = time.time() - start
print("Time to peform reconstruction = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    real_mapped_visibilities = al.util.inversion.mapped_reconstructed_data_from_mapping_matrix_and_reconstruction(
        mapping_matrix=transformed_mapping_matrices[0], reconstruction=reconstruction
    )
diff = time.time() - start
print(
    "Time to compute mapped real visibility reconstruction = {}".format(diff / repeats)
)

start = time.time()
for i in range(repeats):
    imag_mapped_visibilities = al.util.inversion.mapped_reconstructed_data_from_mapping_matrix_and_reconstruction(
        mapping_matrix=transformed_mapping_matrices[1], reconstruction=reconstruction
    )
diff = time.time() - start
print(
    "Time to compute mapped imaginary visibility reconstruction = {}".format(
        diff / repeats
    )
)

start = time.time()
for i in range(repeats):
    al.fit(masked_dataset=masked_interferometer, tracer=tracer)
diff = time.time() - start
print("Time to perform complete fit = {}".format(diff / repeats))

print()
