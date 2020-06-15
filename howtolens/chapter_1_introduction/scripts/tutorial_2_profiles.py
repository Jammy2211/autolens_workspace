# %%
"""
__Profiles__

In this example, we'll create a grid of Cartesian (y,x) coordinates and pass it to the 'light_profiles'  module to create images on this grid and the 'mass_profiles' module to create deflection-angle maps on this grid. 
"""

# %%
#%matplotlib inline

import autolens as al
import autolens.plot as aplt

# %%
"""
Lets use the same grid as the previous tutorial (if you skipped that tutorial, I recommend you go back to it!)
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.05, sub_size=2)

# %%
"""
Next, lets create a *LightProfile* using the 'light_profiles' module, which in PyAutoLens is imported as 'lp' for 
conciseness. We'll use a Sersic function, which is an analytic function often use to depict galaxies.
"""

# %%
sersic_light_profile = al.lp.EllipticalSersic(
    centre=(0.0, 0.0),
    elliptical_comps=(0.0, 0.111111),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=2.5,
)

# %%
"""
We can print a profile to confirm its parameters.
"""

# %%
print(sersic_light_profile)

# %%
"""
We can pass a grid to a *LightProfile* to compute its intensity at every grid coordinate. When we compute an array from 
a grid using a '_from_grid' method like the one below, we have two options for how the calculation is performed.
"""

# %%
light_image = sersic_light_profile.image_from_grid(grid=grid)

# %%
"""
Much like the grids in the previous tutorials, the arrays PyAutoLens computes from these methods are accessible in 
both 2D and 1D.
"""

# %%
print(light_image.shape_2d)
print(light_image.shape_1d)
print(light_image.in_2d[0, 0])
print(light_image.in_1d[0])
print(light_image.in_2d)
print(light_image.in_1d)

# %%
"""
The values computed (e.g. the image) are calculated on the sub-grid and the returned values are stored on the sub-grid, 
which in this case is a 200 x 200 grid.
"""

# %%
print(light_image.sub_shape_2d)
print(light_image.sub_shape_1d)
print(light_image.in_2d[0, 0])
print(light_image[0])

# %%
"""
The benefit of storing all the values on the sub-grid, is that we can now use these values to bin-up the regular grid's 
shape by taking the mean of each intensity value computed on the sub-grid. This ensures that aliasing effects due to 
computing intensities at only one pixel coordinate inside a full pixel do not degrade the image we create.
"""

# %%
print("intensity of top-left grid pixel:")
print(light_image.in_2d_binned[0, 0])
print(light_image.in_1d_binned[0])

# %%
"""
If you find these 2D and 1D arrays confusing - I wouldn't worry about it. From here on, we'll pretty much just use 
these arrays as they returned to us from functions and not think about if they should be in 2D or 1D. Nevertheless, its 
important that you understand PyAutoLens offers these 2D and 1D representations - as it'll help us later when we cover 
fititng lens data!

We can use a profile plotter to plot this image.
"""

# %%
aplt.LightProfile.image(light_profile=sersic_light_profile, grid=grid)

# %%
"""
To perform ray-tracing, we need to create a '*MassProfile*' from the *MassProfile*s module, which we import as mp for 
conciseness. A *MassProfile* is an analytic function that describes the distribution of mass in a galaxy, and therefore 
can be used to derive its surface-density, gravitational potential and most importantly, its deflection angles. For 
those unfamiliar with lensing, the deflection angles describe how light is bent by the *MassProfile* due to the 
curvature of space-time.
"""

# %%
sis_mass_profile = al.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6)

print(sis_mass_profile)

# %%
"""
Just like above, we can pass a grid to a *MassProfile* to compute its deflection angles. These are returned as the grids 
we used in the previous tutorials, so have full access to the 2D / 1D methods and mappings. And, just like the image 
above, they are computed on the sub-grid, so that we can bin up their values to compute more accurate deflection angles.

(If you are new to gravitiational lensing, and are unclear on what a 'deflection-angle' means or what it is used for, 
then I'll explain all in tutorial 4 of this chapter. For now, just look at the pretty pictures they make, and worry 
about what they mean in tutorial 4!).
"""

# %%
mass_profile_deflections = sis_mass_profile.deflections_from_grid(grid=grid)

print("deflection-angles of grid sub-pixel 0:")
print(mass_profile_deflections.in_2d[0, 0])
print("deflection-angles of grid sub-pixel 1:")
print(mass_profile_deflections.in_2d[0, 1])
print()
print("deflection-angles of grid pixel 0:")
print(mass_profile_deflections.in_2d_binned[0, 1])
print()
print("deflection-angles of central grid pixels:")
print(mass_profile_deflections.in_2d_binned[49, 49])
print(mass_profile_deflections.in_2d_binned[49, 50])
print(mass_profile_deflections.in_2d_binned[50, 49])
print(mass_profile_deflections.in_2d_binned[50, 50])

# %%
"""
A profile plotter can plot these deflection angles.

(The black line is the 'critical curve' of the *MassProfile*. We'll cover what this in a later tutorial.)
"""

# %%
aplt.MassProfile.deflections_y(mass_profile=sis_mass_profile, grid=grid)
aplt.MassProfile.deflections_x(mass_profile=sis_mass_profile, grid=grid)

# %%
"""
*MassProfile*s have a range of other properties that are used for lensing calculations, a couple of which we've plotted 
images of below:

Convergence - The surface mass density of the *MassProfile* in dimensionless unit_label which are convenient for lensing 
calcuations.
Potential - The gravitational of the *MassProfile* again in convenient dimensionless unit_label.
Magnification - Describes how much brighter each image-pixel appears due to focusing of light rays by the *MassProfile*.

Extracting arrays of these quantities fom PyAutoLens is exactly the same as for the image and deflection angles above.
"""

# %%
mass_profile_convergence = sis_mass_profile.convergence_from_grid(grid=grid)

mass_profile_potential = sis_mass_profile.potential_from_grid(grid=grid)

mass_profile_magnification = sis_mass_profile.magnification_from_grid(grid=grid)

# %%
"""
Plotting them is equally straight forward.
"""

# %%
aplt.MassProfile.convergence(mass_profile=sis_mass_profile, grid=grid)

aplt.MassProfile.potential(mass_profile=sis_mass_profile, grid=grid)

aplt.MassProfile.magnification(mass_profile=sis_mass_profile, grid=grid)

# %%
"""
Congratulations, you've completed your second PyAutoLens tutorial! Before moving on to the next one, experiment with 
PyAutoLens by doing the following:

1) Change the *LightProfile*'s effective radius and Sersic index - how does the image's appearance change?
2) Change the *MassProfile*'s einstein radius - what happens to the deflection angles, potential and convergence?
3) Experiment with different *LightProfile*s and *MassProfile*s in the light_profiles and mass_profiles modules. 
In particular, use the EllipticalIsothermal profile to introduce ellipticity into a *MassProfile*.
"""
