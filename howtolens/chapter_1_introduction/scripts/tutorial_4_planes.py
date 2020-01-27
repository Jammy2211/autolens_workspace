import autolens as al
import autolens.plot as aplt

# We've learnt how to make galaxy objects out of light and mass profiles. Now, we'll use these galaxies to make a
# strong-gravitational lens.
#
# For newcomers to lens, a strong gravitation lens is a system where two (or more) galaxies align perfectly down
# our line of sight, such that the foreground galaxy's mass (represented as mass profiles) deflects the light
# (represented as light profiles) of the background source galaxy(s).

# When the alignment is just right and the lens is just massive enough, the background source galaxy appears multiple
# times. The schematic below shows a crude drawing of such a system, where two light-rays from the source are bending
# around the lens galaxy and into the observer (light should bend 'smoothly', but drawing this on a keyboard wasn't
# possible - so just pretend the diagonal lines coming from the observer and source are less jagged):

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

# As an observer, we don't see the source's true appearance (e.g. a round blob of light). Instead, we only observe
# its light after it is deflected and lensed by the foreground galaxy's mass. In this exercise, we'll make a source
# galaxy image whose light has been deflected by a lens galaxy.

# In the schematic above, we used the terms 'Image-Plane' and 'Source-Plane'. In lens speak, a 'plane' is a
# collection of galaxies at the same redshift (that is, parallel to one another down our line-of-sight). Therefore:

# - If two or more lens galaxies are at the same redshift in the image-plane, they deflect light in the same way.
#   This means we can sum the convergences, potentials and deflection angles of their mass-profiles.

# - If two or more source galaxies are at the same redshift in the source-plane, their light is ray-traced in the
#   same way. Therefore, when determining their lensed images, we can sum the lensed images of each galaxy's light-profiles.

# So, lets do it - lets use the 'plane' module in AutoLens to create a strong lens system like the one pictured
# above. For simplicity, we'll assume 1 lens galaxy and 1 source galaxy.

# As always, we need a grid, where our grid is the coordinates we 'trace' from the image-plane to the source-plane
# in the lens configuration above. Our grid is therefore no longer just a 'grid', but an image-plane grid
# representing our image-plane coordinates. Thus, lets name as such.
image_plane_grid = al.grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=1)

# Whereas before we called our galaxy's things like 'galaxy_with_light_profile', lets now refer to them by their role
# in lens, e.g. 'lens_galaxy' and 'source_galaxy'.
mass_profile = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)

lens_galaxy = al.Galaxy(redshift=0.5, mass=mass_profile)

light_profile = al.lp.SphericalSersic(
    centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0, sersic_index=1.0
)

source_galaxy = al.Galaxy(redshift=1.0, light=light_profile)

# Lets setup our image-plane. This plane takes the lens galaxy we made above.
image_plane = al.Plane(galaxies=[lens_galaxy])

# Just like we did with galaxies, we can compute quantities from the plane by passing it a grid.

deflections = image_plane.deflections_from_grid(grid=image_plane_grid)
deflections_y = deflections.in_2d[:, :, 0]
deflections_x = deflections.in_2d[:, :, 1]

print("deflection-angles of planes grid pixel 0:")
print(deflections_y[0, 0])
print(deflections_x[0, 0])

print("deflection-angles of planes grid pixel 1:")
print(deflections_y[0, 1])
print(deflections_x[0, 1])

# Plane plotters exist, which work analogously to profile plotters and galaxy plotters.
aplt.plane.deflections_y(plane=image_plane, grid=image_plane_grid)

aplt.plane.deflections_x(plane=image_plane, grid=image_plane_grid)

# Throughout this chapter, we plotted lots of deflection angles. However, if you arn't familiar with strong
# lens, you probably weren't entirely sure what they are actually used for.
#
# The deflection angles tell us how light is 'lensed' by a lens galaxy. By taking the image-plane coordinates and
# deflection angles, we can subtract the two to determine the source-plane's lensed coordinates, e.g.

# source_plane_coordinates = image_plane_coordinates - image_plane_deflection_angles

# Therefore, we can use our image_plane to 'trace' a grid to the source-plane...
source_plane_grid = image_plane.traced_grid_from_grid(grid=image_plane_grid)
print("Traced source-plane coordinates of grid pixel 0:")
print(source_plane_grid[0, :])
print("Traced source-plane coordinates of grid pixel 1:")
print(source_plane_grid[0, :])

# We can also set up a source plane using our source galaxy.
source_plane = al.Plane(galaxies=[source_galaxy])

# Lets inspect our grids - I bet our source-plane isn't the boring uniform grid we plotted in the first tutorial!
aplt.plane.plane_grid(
    plane=image_plane,
    grid=image_plane_grid,
    plotter=aplt.Plotter(labels=aplt.Labels(title="Image-plane Grid")),
)

aplt.plane.plane_grid(
    plane=source_plane,
    grid=source_plane_grid,
    plotter=aplt.Plotter(labels=aplt.Labels(title="Source-plane Grid")),
)

# # We can zoom in on the 'centre' of the source-plane (remembering the lens galaxy was centred at (0.0", 0.0")
aplt.plane.plane_grid(
    plane=source_plane,
    grid=source_plane_grid,
    axis_limits=[-0.1, 0.1, -0.1, 0.1],
    plotter=aplt.Plotter(labels=aplt.Labels(title="Source-plane Grid")),
)

# We can also plotters both planes next to one another, and highlight specific points on the al. This means we can see
# how different image pixels map to the source-plane (and visa versa). The image-plane grid is input and automatically
# traced to the source plane for the plotters.

# (We are inputting the pixel index's into 'points' - the first set of points go from 0 -> 50, which is the top row of
# the image-grid running from the left - as we said it would!)

aplt.plane.image_and_source_plane_subplot(
    image_plane=image_plane,
    source_plane=source_plane,
    grid=image_plane_grid,
    indexes=[
        range(0, 50),
        range(500, 550),
        [1350, 1450, 1550, 1650, 1750, 1850, 1950, 2050, 2150, 2250],
        [6250, 8550, 8450, 8350, 8250, 8150, 8050, 7950, 7850, 7750],
    ],
)

# Clearly, the source-plane's grid is very different to the image-planes! It's not uniform and its centrainly
# not boring!

# We can now ask the question - 'what does our source-galaxy look like in the image-plane'? That is, to us, the
# observer on Earth, how does the source-galaxy appear after lens?. To do this, we simple trace the source
# galaxy's light 'back' from the source-plane grid.
aplt.plane.profile_image(plane=source_plane, grid=source_plane_grid)

# It's a rather spectacular ring of light, but why is it a ring? Well:

# - Our lens galaxy was centred at (0.0", 0.0").
# - Our source-galaxy was centred at (0.0", 0.0").
# - Our lens galaxy had a spherical mass-profile.
# - Our source-galaxy a spherical light-profile.
#
# Given the perfect symmetry of the system, every path the source's light takes around the lens galaxy is
# radially identical. Thus, nothing else but a ring of light can form!

# This is called an 'Einstein Ring' and its radius is called the 'Einstein Radius', which are both named after the
# man who famously used gravitational lens to prove his theory of general relativity.

# Because we know our source-galaxy's light profile, we can also plotters its 'plane-image'. This image is how
# the source intrinsically appears in the source-plane (e.g. without lens). This is a useful thing to know, because
# the source-s light is highly magnified, meaning astronomers can study it in a lot more detail than would
# otherwise be possible!
aplt.plane.plane_image(
    plane=source_plane, grid=source_plane_grid, include=aplt.Include(grid=True)
)

# Plotting the grid over the plane image obscures its appearance, which isn't ideal. We can of course tell
# PyAutoLens not to plot the grid.
aplt.plane.plane_image(
    plane=source_plane, grid=source_plane_grid, include=aplt.Include(grid=False)
)

# For mass profiles, you can also plotters their 'critical curve' and 'caustics', which for those unfamiliar with lens
# are defined as follows:

# Critical Curve - Lines of infinite magnification where the mass profile perfectly 'focuses' light rays. Source light
#                  near a critical curve appears much brighter than its true luminosity!

# Caustic - Given the deflection angles of the mass profile at the critical curves, the caustic is where the critical
#           curve 'maps' too.

aplt.plane.image_and_source_plane_subplot(
    image_plane=image_plane,
    source_plane=source_plane,
    grid=image_plane_grid,
    indexes=[
        range(0, 50),
        range(500, 550),
        [1550, 1650, 1750, 1850, 1950, 2050],
        [8450, 8350, 8250, 8150, 8050, 7950],
    ],
    include=aplt.Include(critical_curves=True, caustics=True),
)

# The critical curves appear on the left hand image, the caustic on the right. Interestingly, there is one curve
# on each image - this may be what you'd expect, but it is actually only true for spherical mass profile, as we'll
# see next.

# We can plot the critical curves and caustics for any arrays image by supplying these bools to the plotter
aplt.plane.convergence(
    plane=image_plane,
    grid=image_plane_grid,
    include=aplt.Include(critical_curves=True, caustics=True),
)

aplt.plane.image_and_source_plane_subplot(
    image_plane=image_plane,
    source_plane=source_plane,
    grid=image_plane_grid,
    include=aplt.Include(critical_curves=False, caustics=False),
)

# And, we're done. This is the first tutorial covering strong-lens and I highly recommend you take a moment
# to really mess about with the code above to see what sort of lensed images you can form. Pay attention to the
# source-plane grid - its appearance can change a lot!
#
# In particular, try:

# 1) Changing the lens galaxy's einstein radius - what happens to the source-plane's image?

# 2) Change the SphericalIsothermal mass-profile to an EllipticalIsothermal mass-profile and set its axis_ratio to 0.8.
#    What happens to the number of source images?

# 3) As discussed at the beginning, planes can be composed of multiple galaxies. Make an the image-plane with two
#    galaxies and see how multi-galaxy lens leads to crazy source images. Also try making a source-plane
#    with two galaxies!

# Finally, if you are a newcomer to strong lens, it might be worth reading briefly about some strong lens theory.
# Don't worry about maths, and equations, and anything scary, but you should read up on:

# - More about what a critical line is.

# - More about what a caustic is.

# - What determines the image multiplicity of the lensed source.
