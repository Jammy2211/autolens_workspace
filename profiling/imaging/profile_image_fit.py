import autolens as al

import time

from profiling.imaging.simulator import simulate_util

repeats = 10

print("Number of repeats = " + str(repeats))
print()

sub_size = 4
radius = 3.0
psf_shape_2d = (21, 21)

print("sub grid size = " + str(sub_size))
print("circular mask radius = " + str(radius) + "\n")
print("psf shape = " + str(psf_shape_2d) + "\n")

lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=al.lp.EllipticalSersic(
        centre=(0.0, 0.0),
        axis_ratio=0.9,
        phi=45.0,
        intensity=0.5,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
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

for data_resolution in ["lsst", "euclid", "hst", "hst_up", "ao"]:

    imaging = simulate_util.load_test_imaging(
        data_type="lens_sie__source_smooth",
        data_resolution=data_resolution,
        psf_shape_2d=psf_shape_2d,
    )

    mask = al.mask.circular(
        shape_2d=imaging.shape_2d,
        pixel_scales=imaging.pixel_scales,
        sub_size=sub_size,
        radius=radius,
    )

    masked_imaging = al.masked.imaging(imaging=imaging, mask=mask)

    print("Light profile fit run times for image type " + data_resolution + "\n")
    print("Number of points = " + str(masked_imaging.grid.sub_shape_1d) + "\n")

    start_overall = time.time()

    start = time.time()
    for i in range(repeats):
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
        profile_image = tracer.profile_image_from_grid(grid=masked_imaging.grid)
    diff = time.time() - start
    print("Time to create profile image = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        blurring_profile_image = tracer.profile_image_from_grid(
            grid=masked_imaging.blurring_grid
        )
    diff = time.time() - start
    print("Time to create blurring profile image = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        blurred_profile_image = masked_imaging.convolver.convolved_image_from_image_and_blurring_image(
            image=profile_image, blurring_image=blurring_profile_image
        )
    diff = time.time() - start
    print("Time to perform PSF Convolution = {}".format(diff / repeats))

    start = time.time()
    for i in range(repeats):
        tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
        al.fit(masked_dataset=masked_imaging, tracer=tracer)
    diff = time.time() - start
    print("Time to perform complete fit = {}".format(diff / repeats))

    print()
