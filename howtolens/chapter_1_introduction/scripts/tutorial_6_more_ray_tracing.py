from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.model.galaxy import galaxy as g
from autolens.lens import ray_tracing
from autolens.array import grids
from autolens.model.galaxy.plotters import galaxy_plotters
from autolens.lens.plotters import plane_plotters
from autolens.lens.plotters import ray_tracing_plotters

from astropy import cosmology

# In this example, we'll reinforce what we learnt about ray-tracing in the last tutorial and introduce the following
# new concepts:

# 1) Short-hand notation for setting up profiles and galaxies, to make code cleaner and easier to read.
# 2) That a tracer can be given any number of galaxies.
# 3) That by specifying redshifts and a cosmology, our results are converted to physical units of kiloparsecs (kpc).

# To begin, lets setup the grids we 'll ray-trace using. Lets do something crazy, and use a
# higher resolution grid then before and set the sub grid size to 4x4 per pixel!

# Lets also stop calling it the 'image_plane_grid', and just remember from now on our grid is in the image-plane.
grid = grids.Grid.from_shape_pixel_scale_and_sub_grid_size(
    shape=(200, 200), pixel_scale=0.025, sub_grid_size=4
)

# Every pixel is sub-gridded by 4x4, so the sub-grid has x16 more coordinates.
print(grid.shape)

# Next, lets setup a lens galaxy. In the previous tutorial, we set up each profile one line at a time. This made
# code long and cumbersome to read. This time we'll setup easy galaxy using one block of code.

# To help us, we've imported the 'light_profiles' and 'mass_profiles' modules as 'lp' and 'mp', and the
# 'galaxy' module as 'g'.

# We'll also give the lens galaxy some attributes we didn't in the last tutorial:

# 1) A light-profile, meaning its light will appear in the image.
# 2) An external shear, which accounts for the deflection of light due to line-of-sight structures.
# 3) A redshift, which the tracer will use to convert arc second coordinates to kpc.
lens_galaxy = g.Galaxy(
    redshift=0.5,
    light=lp.SphericalSersic(
        centre=(0.0, 0.0), intensity=2.0, effective_radius=0.5, sersic_index=2.5
    ),
    mass=mp.EllipticalIsothermal(
        centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0, einstein_radius=1.6
    ),
    shear=mp.ExternalShear(magnitude=0.05, phi=45.0),
)

print(lens_galaxy)

# Lets also create a small satellite galaxy nearby the lens galaxy and at the same redshift.
lens_satellite = g.Galaxy(
    redshift=0.5,
    light=lp.SphericalDevVaucouleurs(
        centre=(1.0, 0.0), intensity=2.0, effective_radius=0.2
    ),
    mass=mp.SphericalIsothermal(centre=(1.0, 0.0), einstein_radius=0.4),
)
print(lens_satellite)

# Lets have a quick look at the appearance of our lens galaxy and its satellite
galaxy_plotters.plot_profile_image(galaxy=lens_galaxy, grid=grid, title="Lens Galaxy")
galaxy_plotters.plot_profile_image(
    galaxy=lens_satellite, grid=grid, title="Lens Satellite"
)

# And their deflection angles - note that the satellite doesn't contribute much to the deflections
galaxy_plotters.plot_deflections_y(
    galaxy=lens_galaxy, grid=grid, title="Lens Galaxy Deflections (y)"
)
galaxy_plotters.plot_deflections_y(
    galaxy=lens_satellite, grid=grid, title="Lens Satellite Deflections (y)"
)
galaxy_plotters.plot_deflections_x(
    galaxy=lens_galaxy, grid=grid, title="Lens Galalxy Deflections (x)"
)
galaxy_plotters.plot_deflections_x(
    galaxy=lens_satellite, grid=grid, title="Lens Satellite Deflections (x)"
)

# Now, lets make two source galaxies at redshift 1.0. Lets not use the terms 'light' and 'mass' to setup the light and
# mass profiles. Instead, lets use more descriptive names of what we think each component represents ( e.g. a 'bulge' and 'disk').
source_galaxy_0 = g.Galaxy(
    redshift=1.0,
    bulge=lp.SphericalDevVaucouleurs(
        centre=(0.1, 0.2), intensity=0.3, effective_radius=0.3
    ),
    disk=lp.EllipticalExponential(
        centre=(0.1, 0.2), axis_ratio=0.8, phi=45.0, intensity=3.0, effective_radius=2.0
    ),
)

source_galaxy_1 = g.Galaxy(
    redshift=1.0,
    disk=lp.EllipticalExponential(
        centre=(-0.3, -0.5),
        axis_ratio=0.6,
        phi=80.0,
        intensity=8.0,
        effective_radius=1.0,
    ),
)
print(source_galaxy_0)
print(source_galaxy_1)

# Lets look at our source galaxies (before lensing)
galaxy_plotters.plot_profile_image(
    galaxy=source_galaxy_0, grid=grid, title="Source Galaxy 0"
)

galaxy_plotters.plot_profile_image(
    galaxy=source_galaxy_1, grid=grid, title="Source Galaxy 1"
)

# Now lets pass our 4 galaxies to the ray_tracing module, which means the following will occur:

# 1) Using the galaxy redshift's, and image-plane and source-plane will be created with the appopriate galaxies.

# Note that we've also supplied the tracer below with a Planck15 cosmology.
tracer = ray_tracing.Tracer.from_galaxies(
    galaxies=[lens_galaxy, lens_satellite, source_galaxy_0, source_galaxy_1],
    cosmology=cosmology.Planck15,
)

# We can next plot the tracer's profile image, which is compute as follows:
# 1) First, using the image-plane grid, the images of the lens galaxy and its satellite are computed.
# 2) Using the mass-profile's of the lens and satellite, their deflection angles are computed.
# 3) These deflection angles are summed, such that the deflection of light due to every mass-profile and both
#    the lens galaxy and its satellite is computed.
# 4) These deflection angles are used to trace every image-grid coordinate to a source-plane coordinate.
# 5) The image of the source galaxies is computed by ray-tracing their light back to the image-plane.
ray_tracing_plotters.plot_profile_image(tracer=tracer, grid=grid)

# As we did previously, we can extract the grids of each plane and inspect the source-plane grid.

traced_grids = tracer.traced_grids_of_planes_from_grid(grid=grid)

plane_plotters.plot_plane_grid(
    plane=tracer.source_plane, grid=traced_grids[1], title="Source-plane Grid"
)

# We can zoom in on the 'centre' of the source-plane.
plane_plotters.plot_plane_grid(
    plane=tracer.source_plane,
    grid=traced_grids[1],
    axis_limits=[-0.2, 0.2, -0.2, 0.2],
    title="Source-plane Grid",
)

# Lets plot the lensing quantities again. Note that, because we supplied our galaxies with redshifts and
# our tracer with a cosmology, our units have been converted to kiloparsecs!
# (This line can take a bit of time to run)
ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer, grid=grid)

# In the previous example, we saw that the tracer computed quantities we plotted (e.g. convergence, potential, etc.)
# Now we've input a cosmology and galaxy redshifts, the tracer has attributes associated with its cosmology.
print("Image-plane arcsec-per-kpc:")
print(tracer.image_plane.arcsec_per_kpc)
print("Image-plane kpc-per-arcsec:")
print(tracer.image_plane.kpc_per_arcsec)
print("Angular Diameter Distance to Image-plane:")
print(tracer.image_plane.angular_diameter_distance_to_earth_in_units(unit_length="kpc"))

print("Source-plane arcsec-per-kpc:")
print(tracer.source_plane.arcsec_per_kpc)
print("Source-plane kpc-per-arcsec:")
print(tracer.source_plane.kpc_per_arcsec)
print("Angular Diameter Distance to Source-plane:")
print(
    tracer.source_plane.angular_diameter_distance_to_earth_in_units(unit_length="kpc")
)

print("Angular Diameter Distance From Image To Source Plane:")
print(
    tracer.angular_diameter_distance_from_image_to_source_plane_in_units(
        unit_length="kpc"
    )
)
print("Lensing Critical convergence:")
print(
    tracer.critical_surface_density_between_planes_in_units(i=0, j=1, unit_length="kpc")
)

# And with that, we've completed tutorial 6. Try the following:

# 1) By changing the lens and source galaxy redshifts, does the image of the tracer change at all?

# 2) What happens to the cosmological quantities as you change these redshifts? Do you remember enough of your
#    cosmology lectures to predict how quantities like the angular diameter distance change as a function of redshift?

# 3) The tracer has a small delay in being computed, whereas other tracers were almost instant. What do you think is
#    the cause of this slow-down?
