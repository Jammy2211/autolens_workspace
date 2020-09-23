# %%
"""
__Example: Lensing__

When two galaxies are aligned perfectly down the line-of-sight to Earth, the background galaxy`s light is bent by the
intervening mass of the foreground galaxy. Its light can be fully bent around the foreground galaxy, traversing multiple
paths to the Earth, meaning that the background galaxy is observed multiple times. This by-chance alignment of two
galaxies is called a strong gravitational lens and a two-dimensional scheme of such a system is pictured below.

PyAutoLens is software designed for modeling these strong lensing systems! To begin, lets import autolens and the plot
module.
"""

# %%
from astropy import cosmology as cosmo
import autolens as al
import autolens.plot as aplt

# %%
"""
To describe the deflection of light, **PyAutoLens** uses *grid* data structures, which are two-dimensional
Cartesian grids of (y,x) coordinates. Below, we make and plot a uniform Cartesian grid:
"""

# %%
grid = al.Grid.uniform(
    shape_2d=(50, 50),
    pixel_scales=0.1,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

# %%
"""
Our aim is to ray-trace this grid`s coordinates to calculate how the lens galaxy`s mass deflects the source galaxy`s
light. We therefore need analytic functions representing light and mass distributions. For this, **PyAutoLens** uses
*Profile* objects and below we use the elliptical `EllipticalSersic` `LightProfile`.object to represent a light distribution:
"""

# %%
sersic_light_profile = al.lp.EllipticalSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.2, 0.1),
    intensity=0.005,
    effective_radius=2.0,
    sersic_index=4.0,
)

# %%
"""
By passing this profile a grid, we can evaluate the light at every coordinate on that grid and create an image
of the `LightProfile`.
"""

# %%
image = sersic_light_profile.image_from_grid(grid=grid)

# %%
"""
The plot module provides convenience methods for plotting properties of objects, like the image of a `LightProfile`.
"""

# %%
aplt.LightProfile.image(light_profile=sersic_light_profile, grid=grid)

# %%
"""
**PyAutoLens** uses `MassProfile` objects to represent different mass distributions and use them to perform ray-tracing
calculations. Below we create an elliptical isothermal `MassProfile` and compute its convergence, gravitational
potential and deflection angles on our Cartesian grid:
"""

# %%
isothermal_mass_profile = al.mp.EllipticalIsothermal(
    centre=(0.0, 0.0), elliptical_comps=(0.1, 0.0), einstein_radius=1.6
)

convergence = isothermal_mass_profile.convergence_from_grid(grid=grid)
potential = isothermal_mass_profile.potential_from_grid(grid=grid)
deflections = isothermal_mass_profile.deflections_from_grid(grid=grid)

# %%
"""
Lets plot the `MassProfile``s convergence, potential and deflection angle map
"""

# %%
aplt.MassProfile.convergence(mass_profile=isothermal_mass_profile, grid=grid)
aplt.MassProfile.potential(mass_profile=isothermal_mass_profile, grid=grid)
aplt.MassProfile.deflections_y(mass_profile=isothermal_mass_profile, grid=grid)
aplt.MassProfile.deflections_x(mass_profile=isothermal_mass_profile, grid=grid)

# %%
"""
For anyone not familiar with gravitational lensing, don`t worry about what the convergence and potential are. The key
thing to note is that the deflection angles describe how a given mass distribution deflections light-rays, which allows
us create strong lens systems like the one shown above!

In **PyAutoLens**, a *Galaxy* object is a collection of `LightProfile` and `MassProfile` objects at a given redshift.
The code below creates two galaxies representing the lens and source galaxies shown in the strong lensing diagram above.
"""

# %%
lens_galaxy = al.Galaxy(
    redshift=0.5, light=sersic_light_profile, mass=isothermal_mass_profile
)

source_light_profile = al.lp.EllipticalExponential(
    centre=(0.3, 0.2), elliptical_comps=(0.1, 0.0), intensity=0.05, effective_radius=0.5
)

source_galaxy = al.Galaxy(redshift=1.0, light=source_light_profile)

# %%
"""
The geometry of the strong lens system depends on the cosmological distances between the Earth, lens and source and
therefore the redshifts of the lens galaxy and source galaxy objects. By passing these *Galaxy* objects to the
*Tracer* class **PyAutoLens** uses these galaxy redshifts and a cosmological model to create the appropriate strong
lens system.
"""

# %%
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy], cosmology=cosmo.Planck15
)

# %%
"""
When computing the image from the tracer above, the tracer performs all ray-tracing for the given strong lens system.
This includes using the lens galaxy`s `MassProfile` to deflect the light-rays that are traced to the source galaxy.
This makes the image below, where the source`s light appears as a multiply imaged and strongly lensed Einstein ring.
"""

# %%
image = tracer.image_from_grid(grid=grid)

aplt.Tracer.image(tracer=tracer, grid=grid)

# %%
"""
The Tracer plotter includes the `MassProfile` quantities we plotted previous. subplot that plots all these quantities simultaneously.
"""
aplt.Tracer.convergence(tracer=tracer, grid=grid)
aplt.Tracer.potential(tracer=tracer, grid=grid)
aplt.Tracer.deflections_y(tracer=tracer, grid=grid)
aplt.Tracer.deflections_x(tracer=tracer, grid=grid)

# %%
"""
It also includes a subplot that plots all these quantities simultaneously.
"""
aplt.Tracer.subplot_tracer(tracer=tracer, grid=grid)

# %%
"""
The *Tracer* is composed of planes, for the system above just two planes, an image-plane (at redshift=0.5) and a 
source-plane (at redshift=1.0). When creating the image from a Tracer, the `MassProfile` is used to `ray-trace` the 
image-plane grid to the source-plane grid, via the `MassProfile``s deflection angles.

We can use the Tracer`s traced_grid method to plot the image-plalne and source-plane grids.
"""
traced_grids = tracer.traced_grids_of_planes_from_grid(grid=grid)
aplt.Grid(grid=traced_grids[0])  # Image-plane grid.
aplt.Grid(grid=traced_grids[1])  # Source-plane grid.

# %%
"""
The PyAutoLens API has been designed such that all of the objects introduced above are extensible. *Galaxy* objects can
take many profiles and *Tracer* objects many galaxies. If the galaxies are at different redshifts a strong lensing
system with multiple lens planes will be created, performing complex multi-plane ray-tracing calculations.

To finish, lets create a tracer using 3 galaxies at different redshifts. The mass distribution of the first lens
galaxy has separate components for its stellar mass and dark matter. This forms a system with two distinct Einstein
rings!
"""

# lens_galaxy_0 = al.Galaxy(
#     redshift=0.5,
#     bulge=al.lmp.EllipticalSersic(
#         centre=(0.0, 0.0),
#         elliptical_comps=(0.0, 0.05),
#         intensity=0.5,
#         effective_radius=0.3,
#         sersic_index=2.5,
#         mass_to_light_ratio=0.3,
#     ),
#     disk=al.lmp.EllipticalExponential(
#         centre=(0.0, 0.0),
#         elliptical_comps=(0.0, 0.1),
#         intensity=1.0,
#         effective_radius=2.0,
#         mass_to_light_ratio=0.2,
#     ),
#     dark=al.mp.SphericalNFW(centre=(0.0, 0.0), kappa_s=0.08, scale_radius=30.0),
# )
#
# lens_galaxy_1 = al.Galaxy(
#     redshift=1.0,
#     sersic=al.lp.EllipticalExponential(
#         centre=(0.1, 0.1),
#         elliptical_comps=(0.05, 0.1),
#         intensity=3.0,
#         effective_radius=0.1,
#     ),
#     mass=al.mp.EllipticalIsothermal(
#         centre=(0.1, 0.1), elliptical_comps=(0.05, 0.1), einstein_radius=0.4
#     ),
# )
#
# source_galaxy = al.Galaxy(
#     redshift=2.0,
#     sersic=al.lp.EllipticalSersic(
#         centre=(0.2, 0.2),
#         elliptical_comps=(0.0, 0.111111),
#         intensity=2.0,
#         effective_radius=0.1,
#         sersic_index=1.5,
#     ),
# )
#
# tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy_0, lens_galaxy_1, source_galaxy])
#
# # %%
# """
# This is what the lens looks like:
# """
#
# # %%
# aplt.Tracer.image(tracer=tracer, grid=grid)
