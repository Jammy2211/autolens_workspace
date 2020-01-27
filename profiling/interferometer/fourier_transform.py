import autolens as al

import numpy as np
import time

repeats = 1

visibilities = 1000
shape_2d = (100, 100)
image_pixels = shape_2d[0] * shape_2d[1]
source_pixels = 1000

shape_data = 8 * visibilities
shape_preloads = visibilities * image_pixels * 2
shape_mapping_matrix = visibilities * source_pixels

total_shape = shape_data + shape_preloads + shape_mapping_matrix

print("Data Memory Use (GB) = " + str(shape_data * 8e-9))
print("PreLoad Memory Use (GB) = " + str(shape_preloads * 8e-9))
print("Mapping Matrix Memory Use (GB) = " + str(shape_mapping_matrix * 8e-9))
print("Total Memory Use (GB) = " + str(total_shape * 8e-9))
print()

# Only delete this if the memory use looks... Okay
# stop

uv_wavelengths = np.ones(shape=(visibilities, 2))
grid = al.grid.uniform(shape_2d=shape_2d, pixel_scales=0.05)
image = al.array.ones(shape_2d=shape_2d, pixel_scales=0.05)

transformer = al.transformer(
    uv_wavelengths=uv_wavelengths, grid_radians=grid, preload_transform=False
)

start = time.time()
for i in range(repeats):
    transformer.real_visibilities_from_image(image=image)
diff = time.time() - start
print("Real Visibilities Time = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    transformer.imag_visibilities_from_image(image=image)
diff = time.time() - start
print("Imaginary Visibilities Time = {}".format(diff / repeats))

transformer = al.transformer(
    uv_wavelengths=uv_wavelengths, grid_radians=grid, preload_transform=True
)

start = time.time()
for i in range(repeats):
    transformer.real_visibilities_from_image(image=image)
diff = time.time() - start
print("Real Visibilities PreLoad Time = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    transformer.imag_visibilities_from_image(image=image)
diff = time.time() - start
print("Imaginary Visibilities PreLoad Time = {}".format(diff / repeats))
