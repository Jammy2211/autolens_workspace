"""
Tutorial 5: More Ray Tracing
============================

We'll now reinforce the ideas that we learnt about ray-tracing in the previous tutorial and introduce the following
new concepts:

- What critical curves and caustics are.
 - That by specifying redshifts and a cosmology, the results are converted from arc-second coordinates to physical
 units of kiloparsecs (kpc). Again, if you're not an Astronomer, you may not be familiar with the unit of parsec, it
 may be worth a quick Google!
  - That a `Tracer` can be given any number of galaxies.

Up to now, the planes have also had just one lens galaxy or source galaxy at a time. In this example, the tracer will
have multiple galaxies at each redshift, meaning that each plane has more than one galaxy. In terms of lensing
calculations:

- If two or more lens galaxies are at the same redshift in the image-plane, the convergences, potentials and
deflection angles of their mass profiles are summed when performing lensing calculations.

- If two or more source galaxies are at the same redshift in the source-plane, their light can simply be summed before
ray tracing.

The `Tracer` fully accounts for this.

__Contents__

**Initial Setup:** To begin, lets setup the grid we'll ray-trace using.
**Concise Code:** Lets set up the tracer used in the previous tutorial.
**Critical Curves:** To end, we can finally explain what the black lines that have appeared on many of the plots.
**Caustics:** In the previous tutorial, we plotted the critical curves of the mass profile on the image-plane.
**Units:** Lets plot the lensing quantities again.
**More Complexity:** We now make a lens with some attributes we didn`t in the last tutorial.
**Multi Galaxy Ray Tracing:** Now lets pass our 4 galaxies to a `Tracer`, which means the following will occur.
**Wrap Up:** Summary of the script and next steps.

"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

To begin, lets setup the grid we'll ray-trace using. But, lets do something crazy and use a higher resolution than 
the previous tutorials!

Lets also stop calling it the `image_plane_grid`, and just remember from now on our `grid` is in the image-plane.
"""
grid = al.Grid2D.uniform(shape_native=(250, 250), pixel_scales=0.02)

"""
The grid is now shape 250 x 250, which has more image-pixels than the 100 x 100 grid used previously.
"""
print(grid.shape_native)
print(grid.shape_slim)

"""
__Concise Code__

Lets set up the tracer used in the previous tutorial.

Up to now, we have set up each profile one line at a time, making the code long and cumbersome to read.

From here on, we'll set up galaxies in a single block of code, making it more concise and readable.
"""
lens = al.Galaxy(
    redshift=0.5, mass=al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)
)

source = al.Galaxy(
    redshift=1.0,
    light=al.lp.SersicCoreSph(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=1.0,
        radius_break=0.025,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

"""
__Critical Curves__

To end, we can finally explain what the black lines that have appeared on many of the plots throughout this chapter 
actually are. 

These lines are called the 'critical curves', and they define line of infinite magnification due to a mass profile. 
They therefore mark where in the image-plane a mass profile perfectly `focuses` light rays such that if a source is 
located there, it will appear very bright: potentially 10-100x as brighter than its intrinsic luminosity.

The outer white line is a `tangential_critical_curve`, because it describes how the image of the source galaxy is stretched
tangentially. There is also an inner `radial_critical_curve` which appears in yellow on figures, which describes how the 
image of the source galaxy is stretched radially. 

However, a radial critical curve only appears when the lens galaxy's mass profile is shallower than isothermal (e.g. 
when its inner mass slope is less steep than a steep power-law). To make it appear below, we therefore change
the mass profile of our lens galaxy to a `PowerLawSph` with a slope of 1.8.

In the next tutorial, we'll introduce 'caustics', which are where the critical curves map too in the source-plane.
"""
mass_profile = al.mp.PowerLawSph(centre=(0.0, 0.0), einstein_radius=1.6, slope=1.8)

lens = al.Galaxy(redshift=0.5, mass=mass_profile)

tangential_critical_curve_list = al.LensCalc.from_mass_obj(
    mass_obj=mass_profile
).tangential_critical_curve_list_from(grid=grid)
radial_critical_curves_list = al.LensCalc.from_mass_obj(
    mass_obj=mass_profile
).radial_critical_curve_list_from(grid=grid)


"""
__Caustics__

In the previous tutorial, we plotted the critical curves of the mass profile on the image-plane. We will now plot the
'caustics', which correspond to each critical curve ray-traced to the source-plane. This is computed by using the 
lens galaxy mass profile's to calculate the deflection angles at the critical curves and ray-trace them to the 
source-plane.

As discussed in the previous tutorial, critical curves mark regions of infinite magnification. Thus, if a source
appears near a caustic in the source plane it will appear significantly brighter than its true luminosity. 

We again have to use a mass profile with a slope below 2.0 to ensure a radial critical curve and therefore radial
caustic is formed. As above, the tangential critical curve is white and maps to the tangential caustic in the source-plane,
which is also white. The radial critical curve is yellow and maps to the radial caustic, which is also yellow.

We can plot both the tangential and radial critical curves and caustics using the `lines=`/`positions=` overlays object. The 
critical curves are plotted only for the image-plane grid, whereas the caustic in the source plane.
"""
sis_mass_profile = al.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=1.6)

tracer = al.Tracer(galaxies=[lens, source])

tangential_critical_curve_list = al.LensCalc.from_tracer(
    tracer=tracer
).tangential_critical_curve_list_from(grid=grid)
radial_critical_curves_list = al.LensCalc.from_tracer(
    tracer=tracer
).radial_critical_curve_list_from(grid=grid)


aplt.plot_grid(grid=tracer.traced_grid_2d_list_from(grid=grid)[0], title="Plane 0 Grid")
aplt.plot_grid(grid=tracer.traced_grid_2d_list_from(grid=grid)[1], title="Plane 1 Grid")

tangential_caustic_list = al.LensCalc.from_tracer(
    tracer=tracer
).tangential_caustic_list_from(grid=grid)
radial_caustics_list = al.LensCalc.from_tracer(tracer=tracer).radial_caustic_list_from(
    grid=grid
)


"""
We can also plot the caustic on the source-plane image.
"""
aplt.plot_array(array=tracer.image_2d_list_from(grid=grid)[1], title="Plane 1 Image")

"""
Caustics also mark the regions in the source-plane where the multiplicity of the strong lens changes. That is,
if a source crosses a caustic, it goes from 2 images to 1 image. Try and show this yourself by changing the (y,x) 
centre of the source-plane galaxy's light profile!
"""
source = al.Galaxy(
    redshift=1.0,
    light=al.lp.SersicCoreSph(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens, source])

aplt.plot_array(array=tracer.image_2d_list_from(grid=grid)[1], title="Plane 1 Image")

"""
__Units__

Lets plot the lensing quantities again. However, we'll now use the `Units` object of the **PyAutoLens** plotter module 
to set `in_kpc=True` and therefore plot the y and x axes in kiloparsecs.

This conversion is performed automatically, using the galaxy redshifts and cosmology.
"""

aplt.subplot_tracer(tracer=tracer, grid=grid)
aplt.subplot_galaxies_images(tracer=tracer, grid=grid)

"""
If you're too familiar with Cosmology, it will be unclear how exactly we converted the distance units from 
arcseconds to kiloparsecs. You'll need to read up on your Cosmology lecture to understand this properly.

You can create a `Cosmology` object, which provides many methods for calculation different cosmological quantities, 
which are shown  below (if you're not too familiar with cosmology don't worry that you don't know what these mean, 
it isn't massively important for using **PyAutoLens**).

We will use a flat lambda CDM cosmology, which is the standard cosmological model often assumed in scientific studies.
"""
cosmology = al.cosmo.FlatLambdaCDM(H0=70, Om0=0.3)

print("Image-plane arcsec-per-kpc:")
print(cosmology.arcsec_per_kpc_from(redshift=0.5))
print("Image-plane kpc-per-arcsec:")
print(cosmology.kpc_per_arcsec_from(redshift=0.5))
print("Angular Diameter Distance to Image-plane (kpc):")
print(cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=0.5))

print("Source-plane arcsec-per-kpc:")
print(cosmology.arcsec_per_kpc_from(redshift=1.0))
print("Source-plane kpc-per-arcsec:")
print(cosmology.kpc_per_arcsec_from(redshift=1.0))
print("Angular Diameter Distance to Source-plane:")
print(cosmology.angular_diameter_distance_to_earth_in_kpc_from(redshift=1.0))

print("Angular Diameter Distance From Image To Source Plane:")
print(
    cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=0.5, redshift_1=1.0
    )
)
print("Lensing Critical convergence:")
print(
    cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=0.5, redshift_1=1.0
    )
)

"""
__More Complexity__

We now make a lens with some attributes we didn`t in the last tutorial:

 - A light profile representing a `bulge` of stars, meaning the lens galaxy's light will appear in the image for the
 first time.
 - An external shear, which accounts for the deflection of light due to line-of-sight structures.
"""
lens = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.SersicSph(
        centre=(0.0, 0.0), intensity=2.0, effective_radius=0.5, sersic_index=2.5
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=1.6
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.0),
)

print(lens)

"""
Lets also create a small satellite galaxy nearby the lens galaxy and at the same redshift.
"""
lens_satellite = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.DevVaucouleursSph(
        centre=(1.0, 0.0), intensity=2.0, effective_radius=0.2
    ),
    mass=al.mp.IsothermalSph(centre=(1.0, 0.0), einstein_radius=0.4),
)

print(lens_satellite)

"""
Lets have a quick look at the appearance of our lens galaxy and its satellite.
"""


"""
And their deflection angles, noting that the satellite does not contribute as much to the deflections.
"""


# NOTE: In the new API, pass title directly as a string:
# title="Lens Galaxy Deflections (x)"


"""
Now, lets make two source galaxies at redshift 1.0. Instead of using the name `light` for the light profiles, lets 
instead use more descriptive names that indicate what morphological component of the galaxy the light profile 
represents. In this case, we'll use the terms `bulge` and `disk`, the two main structures that a galaxy can be made of
"""
source_0 = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.DevVaucouleursSph(
        centre=(0.1, 0.2), intensity=0.3, effective_radius=0.3
    ),
    disk=al.lp.ExponentialCore(
        centre=(0.1, 0.2),
        ell_comps=(0.111111, 0.0),
        intensity=3.0,
        effective_radius=2.0,
    ),
)

source_1 = al.Galaxy(
    redshift=1.0,
    disk=al.lp.ExponentialCore(
        centre=(-0.3, -0.5),
        ell_comps=(0.1, 0.0),
        intensity=8.0,
        effective_radius=1.0,
    ),
)

print(source_0)
print(source_1)

"""
Lets look at our source galaxies (before lensing)
"""


"""
__Multi Galaxy Ray Tracing__

Now lets pass our 4 galaxies to a `Tracer`, which means the following will occur:

 - Using the galaxy redshift`s, and image-plane and source-plane will be created each with two galaxies galaxies.

We've also pass the tracer below a Planck15 cosmology, where the cosomology of the Universe describes exactly how 
ray-tracing is performed.
"""
tracer = al.Tracer(
    galaxies=[lens, lens_satellite, source_0, source_1],
    cosmology=al.cosmo.Planck15(),
)

"""
We can now plot the tracer`s image, which now there are two galaxies in each plane is computed as follows:

 1) First, using the image-plane grid, the images of the lens galaxy and its satellite are computed.

 2) Using the mass profiles of the lens and its satellite, their deflection angles are computed.

 3) These deflection angles are summed, such that the deflection of light due to the mass profiles of both galaxies in 
 the image-plane is accounted for.

 4) These deflection angles are used to trace every image-grid coordinate to the source-plane.

 5) The image of the source galaxies is computed by summing both of their images and ray-tracing their light back to 
 the image-plane.
 
This process is pretty much the same as we have single in previous tutorials when there is one galaxy per plane. We
are simply summing the images and deflection angles of the galaxies before using them to perform ray-tracing.
"""
aplt.plot_array(array=tracer.image_2d_from(grid=grid), title="Image")

"""
As we did previously, we can plot the source plane grid to see how each coordinate was traced. 
"""
aplt.plot_grid(grid=tracer.traced_grid_2d_list_from(grid=grid)[1], title="Plane 1 Grid")

"""
We can zoom in on the source-plane to reveal the inner structure of the caustic.
"""

aplt.plot_grid(grid=tracer.traced_grid_2d_list_from(grid=grid)[1], title="Plane 1 Grid")

"""
__Wrap Up__

Tutorial 6 completed! Try the following:

 1) If you change the lens and source galaxy redshifts, does the tracer's image change?

 2) What happens to the cosmological quantities as you change these redshifts? Do you remember enough of your 
 cosmology lectures to predict how quantities like the angular diameter distance change as a function of redshift?

 3) The tracer has a small delay in being computed, whereas other tracers were almost instant. What do you think 
 is the cause of this slow-down?
"""
