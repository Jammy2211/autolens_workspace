import autolens as al

import numpy as np
import time

repeats = 5

total_visibilities = 50000

real_space_shape_2d = (151, 151)
real_space_pixel_scales = 0.05
real_space_sub_size = 1
real_space_radius = 3.0

image_pixels = real_space_shape_2d[0] * real_space_shape_2d[1]

shape_data = 8 * total_visibilities
shape_preloads = total_visibilities * image_pixels * 2

total_shape = shape_data + shape_preloads

print("Data Memory Use (GB) = " + str(shape_data * 8e-9))
print("PreLoad Memory Use (GB) = " + str(shape_preloads * 8e-9))
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

real_space_mask = al.mask.circular(
    shape_2d=real_space_shape_2d,
    pixel_scales=real_space_pixel_scales,
    sub_size=real_space_sub_size,
    radius=real_space_radius,
)

masked_interferometer = al.masked.interferometer(
    interferometer=interferometer,
    real_space_mask=real_space_mask,
    visibilities_mask=np.full(fill_value=False, shape=total_visibilities),
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
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    profile_image = tracer.profile_image_from_grid(grid=masked_interferometer.grid)
diff = time.time() - start
print("Time to create profile image = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    visibilities = tracer.profile_visibilities_from_grid_and_transformer(
        grid=masked_interferometer.grid, transformer=masked_interferometer.transformer
    )
diff = time.time() - start
print("Time to create profile visibilities = {}".format(diff / repeats))

start = time.time()
for i in range(repeats):
    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
    al.fit(masked_dataset=masked_interferometer, tracer=tracer)
diff = time.time() - start
print("Time to perform complete fit = {}".format(diff / repeats))

print()
