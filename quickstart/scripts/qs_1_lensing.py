import autolens as al

### LENSING ####

# Strong lens modeling uses grids of (y,x) coordinates (e.g. in arc-seconds) to trace light-rays that are deflected by
# a strong lens galaxy. Before light is deflected, the grid of coordinates is the 'image-plane' grid.

# To begin, we make an image-plane grid with PyAutoLens. The grid below grid consists of 100 x 100 coordinates and has
# a pixel-to-arcsecond conversion scale of 0.05".

grid = al.Grid.from_shape_pixel_scale_and_sub_grid_size(
    shape=(100, 100), pixel_scale=0.05, sub_grid_size=1
)

al.grid_plotters.plot_grid(grid=grid, title="Image-Plane Uniform Grid")

# To perform ray-tracing, we create a 'mass-profile'. A mass-profile is an analytic function that describes a
# distribution of mass and is used to derive its convergence, gravitational potential and most importantly its
# deflection angles, which describe how light is bent by the mass-profile's curvature of space-time.

sis_mass_profile = al.mass_profiles.EllipticalIsothermal(
    centre=(0.0, 0.0), axis_ratio=0.9, phi=45.0, einstein_radius=1.6
)

mass_profile_deflections = sis_mass_profile.deflections_from_grid(grid=grid)

# The deflection angles trace the (y,x) grid from the image-plane to the source-plane (where the source appears
# unlensed). We can subtract the deflection angles from the image-plane grid to get the source-plane grid.

source_plane_grid = grid - mass_profile_deflections

al.grid_plotters.plot_grid(grid=source_plane_grid, title="Source Plane Lensed Grid")

# We use mass profiles to map between grids and therefore 'trace' light-rays through a strong lens system. Light
# profiles represent the light using analytic profiles (e.g. a Sersic function).

# Below, we evaluate and plot a Sersic light profile on the image-plane grid and lensed source-plane grid. This shows
# how the galaxy's light is deflected by the mass profile above.

sersic_light_profile = al.light_profiles.EllipticalSersic(
    centre=(0.0, 0.0),
    axis_ratio=0.8,
    phi=45.0,
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

al.profile_plotters.plot_image(
    light_profile=sersic_light_profile,
    grid=grid,
    title="Image-Plane Sersic Profile Image",
)

al.profile_plotters.plot_image(
    light_profile=sersic_light_profile,
    grid=source_plane_grid,
    title="Lensed Source-Plane Sersic Profile Image",
)

# We make 'Galaxy' objects from light and mass profiles to perform lensing calculations, where:

# 1) Galaxies are made from multiple light-profiles and / or mass-profiles.
# 2) Quantities like a galaxy's image and deflection angles are computed by summing those of its individual profiles.
# 3) Galaxies have redshifts, defining where they are relative to one another for ray-tracing.

# Below we make a galaxy from light and mass profiles and use this to plot some of its quantities.

light_profile_0 = al.light_profiles.SphericalExponential(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
)

light_profile_1 = al.light_profiles.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=2.0, sersic_index=3.0
)

mass_profile_0 = al.mass_profiles.SphericalIsothermal(
    centre=(0.0, 0.0), einstein_radius=0.3
)

mass_profile_1 = al.mass_profiles.EllipticalIsothermal(
    centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=1.0
)

galaxy = al.Galaxy(
    redshift=0.5,
    light_profile_0=light_profile_0,
    light_profile_1=light_profile_1,
    mass_profile_0=mass_profile_0,
    mass_profile_1=mass_profile_1,
)

al.galaxy_plotters.plot_profile_image(galaxy=galaxy, grid=grid)
al.galaxy_plotters.plot_deflections_y(galaxy=galaxy, grid=grid)
al.galaxy_plotters.plot_deflections_x(galaxy=galaxy, grid=grid)

# To perform ray-tracing we create multiple galaxies at different redshifts. Lets setup the 2-plane strong lens system below:

#  Observer                  Image-Plane               Source-Plane
#  (z=0, Earth)               (z = 0.5)                (z = 1.0)
#
#           ----------------------------------------------
#          /                                              \ <---- This is one of the source's light-rays
#         /                      __                       \
#    o   /                      /  \                      __
#    |  /                      /   \                     /  \
#   /\  \                      \   /                     \__/
#        \                     \__/                 Source Galaxy (s)
#         \                Lens Galaxy(s)                /
#           \                                           / <----- And this is its other light-ray
#            ------------------------------------------/

# We can pass galaxies into a 'Tracer' to create this strong lens system.

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mass_profiles.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.light_profiles.SphericalSersic(
        centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# We can then pass our image-plane grid to the tracer to 'ray-trace' it through the strong lens system.
traced_image = tracer.profile_image_from_grid(grid=grid)
al.ray_tracing_plotters.plot_profile_image(tracer=tracer, grid=grid)

# PyAutoLens has subplot plotters that plot all relevent quantities of an object. For a tracer, the subplot plots its
# traced image, convergence, potential and deflection angles.
al.ray_tracing_plotters.plot_ray_tracing_subplot(tracer=tracer, grid=grid)

# Hopefully you'll agree that performing ray-tracing in PyAutoLens is straight-forward! Before continuing,
# try the following:

# - Change the lens galaxy mass profile to an EllipticalIsothermal - what happens?
# - Add more lens and source galaxies to the tracer.
# - Change the resolution (pixel-scale) of the image-plane grid.
