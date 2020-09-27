# %%
"""
__Example: Point Sources__

PyAutoLens is primarily designed for strongly lensed galaxies, whose extended surface brightness is lensed into the
aweinspiring giant arcs and Einstein rings we see in high quality lens imaging. However, there are a lot of science
cases where the backgound source is not extended but a point-source, for example strongly lensed quasars and supernovae.

For these objects, we do not want to model the source using a `LightProfile` which implicitly assumes an extended
surface brightness distribution. Instead, we assume that our source is a point with centre (y,x). Our ray-tracing
calculations no longer trace light from the source plane to the image-plane, but instead want to find the locations
the point-source multiple image appear in the image-plane.

Finding the multiple images of a mass model given a (y,x) coordinate in the source plane is an iterative problem
performed in a very different way to ray-tracing a `LightProile`. In this example, we introduce **PyAutoLens**`s
_PositionSolver_, which does exactly this and thus makes the analysis of strong lensed quasars, supernovae and
point-like source`s possible in **PyAutoLens**! we'll also show how these tools allow us to compute the flux-ratios
and time-delays of the point-source.
"""


# %%
import autolens as al
import autolens.plot as aplt

# %%
"""
To begin, we will create an image of strong lens using a simple `EllipticalIsothermal` mass model and source with an
_EllipticalExponential_ light profile. Although we are going to show how **PyAutoLens**`s positional analysis tools model
point-sources, showing the tools using an extended source will make it visibly clearer where the multiple images of
the point source are!

Below, we set up a `Tracer` using a `Grid`, `LightProfile`, `MassProfile` and two `Galaxy``.. These objects are 
introduced in the `lensing.py` example script, so if it is unclear what they are doing you should read through that
example first before contuining!
"""

# %%
grid = al.Grid.uniform(
    shape_2d=(100, 100),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

isothermal_mass_profile = al.mp.EllipticalIsothermal(
    centre=(0.001, 0.001), einstein_radius=1.0, elliptical_comps=(0.0, 0.111111)
)

exponential_light_profile = al.lp.EllipticalExponential(
    centre=(0.07, 0.07),
    elliptical_comps=(0.2, 0.0),
    intensity=0.05,
    effective_radius=0.2,
)

lens_galaxy = al.Galaxy(redshift=0.5, mass=isothermal_mass_profile)

source_galaxy = al.Galaxy(redshift=1.0, light=exponential_light_profile)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# %%
"""
Lets plot the image of our strongly lensed source galaxy. By eye, we can clearly see there are four multiple images 
located in a cross configuration, which are the four (y,x) multiple image coordinates we want our positional solver
to find! 
"""

# %%
# aplt.Tracer.image(tracer=tracer, grid=grid)

# %%
"""
Infact, the `Tracer` has the `PositionSolver` we introduce next built into it, and we can use this to plot the
_Tracer_`s multiple images on the figure (they should appear as black dots on the image)!
"""

# %%
# aplt.Tracer.image(tracer=tracer, grid=grid, include=aplt.Include(multiple_images=True))

# %%
"""
At this point, you might be wondering why don't we use the image of the lensed source to compute our multiple images?
Can`t we just find the pixels in the image whose flux is brighter than its neighboring pixels? 

Although this would work, the problem is that for positional modeling we want to know the (y,x) coordinates of the 
multiple images at a significantly higher precision than the `Grid` we are plotting the image on. In this example, 
the `Grid` has a pixel scale of 0.05", however we want to determine our multiple image positions at scales of 0.01"
or less. We could increase our grid resolutin to 0.01" or below, but this will quickly become very computationally
expensive, thus a bespoke `PositionSolver` is required!
"""

solver = al.PositionsFinder(
    grid=grid,
    pixel_scale_precision=0.001,
    upscale_factor=2,
    distance_from_source_centre=0.01,
)

positions = solver.solve(
    lensing_obj=lens_galaxy, source_plane_coordinate=source_galaxy.light.centre
)

aplt.Tracer.image(tracer=tracer, grid=grid, positions=positions)
