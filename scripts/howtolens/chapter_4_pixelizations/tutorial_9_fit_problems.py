"""
Tutorial 9: Fit Problems
========================

To begin, make sure you have read the `introduction` file carefully, as a clear understanding of how the Bayesian
evidence works is key to understanding this chapter!

In the previous chapter we investigated two pixelization's: `Rectangular` and `Delaunay`. We argued that the
latter was better than the former, because it dedicated more source-pixels to the regions of the source-plane where we
had more data, e.g, the high-magnification regions. Therefore, we could fit the data using fewer source pixels,
which improved computational efficiency and increased the Bayesian evidence.

So far, we've used just one regularization scheme; `Constant`. As the name suggests, this scheme applies just one
regularization coefficient when comparing source pixel fluxes to apply smoothing. Here is a recap of our discussion
about regularization from chapter 4:

-------------------------------------------- 

When the inversion reconstructs the source, it does not *only* compute the set of source-pixel fluxes that best-fit
the image. It also regularizes this solution, whereby it goes to every pixel on the rectangular source-plane grid
and computes the different between the reconstructed flux values of every source pixel with its 4 neighboring pixels.
If the difference in flux is large the solution is penalized, reducing its log likelihood. You can think of this as
us applying a 'smoothness prior' on the reconstructed source galaxy's light.

This smoothing adds a 'penalty term' to the log likelihood of an inversion which is the summed difference between the
reconstructed fluxes of every source-pixel pair multiplied by the `coefficient`. By setting the regularization
coefficient to zero, we set this penalty term to zero, meaning that regularization is completely omitted.

Why do we need to regularize our solution? We just saw why, if we do not apply this smoothness prior to the source, we
`over-fit` the image and reconstruct a noisy source with lots of extraneous features. This is what the  large flux
values located at the exterior regions of the source reconstruction above are. If the inversions's sole aim is to
maximize the log likelihood, it can do this by fitting *everything* accurately, including the noise.

----------------------------------------------

When using a `ConstantSplit` regularization scheme, we regularize the source by adding up the difference in fluxes 
between all source-pixels multiplied by one single value of the regularization coefficient. This means that every
single source pixel receives the same `level` of regularization, regardless of whether it is reconstructing the
bright central regions of the source or its faint exterior regions.

In this tutorial, we'll learn why our magnification-based pixelization and constant regularization schemes are not
optimal. We'll inspect fits to three strong lenses, simulated using the same mass profile but with different
sources whose light profiles become gradually more compact. For all 3 fits, we'll use the same source-plane resolution
and a regularization_coefficient that maximize the Bayesian evidence. Thus, these are the `best` source reconstructions
we can hope to achieve when adapting to the magnification.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

"""
__Initial Setup__

we'll use 3 sources whose `effective_radius` and `sersic_index` are changed such that each is more compact that the last.
"""
source_galaxy_flat = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.15),
        intensity=0.2,
        effective_radius=0.5,
        sersic_index=1.0,
    ),
)

source_galaxy_compact = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.15),
        intensity=0.2,
        effective_radius=0.2,
        sersic_index=2.5,
    ),
)

source_galaxy_super_compact = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.15),
        intensity=0.2,
        effective_radius=0.1,
        sersic_index=4.0,
    ),
)

"""
The function below uses each source galaxy to simulate imaging data. It performs the usual tasks we are used to 
seeing (make the PSF, galaxies, tracer, etc.).
"""


def simulate_for_source_galaxy(source_galaxy):
    grid = al.Grid2D.uniform(shape_native=(150, 150), pixel_scales=0.05)

    psf = al.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=0.05, pixel_scales=0.05
    )

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.111111, 0.0), einstein_radius=1.6
        ),
    )

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=100.0,
        add_poisson_noise=True,
        noise_seed=1,
    )

    return simulator.via_tracer_from(tracer=tracer, grid=grid)


"""
__Mask__

we'll use a 3.0" mask to fit all three of our sources.
"""
mask = al.Mask2D.circular(shape_native=(150, 150), pixel_scales=0.05, radius=3.0)

"""
__Simulator__

Now, lets simulate all 3 of our source's as to create `Imaging` data.
"""
dataset_source_flat = simulate_for_source_galaxy(source_galaxy=source_galaxy_flat)

dataset_source_compact = simulate_for_source_galaxy(source_galaxy=source_galaxy_compact)

dataset_source_super_compact = simulate_for_source_galaxy(
    source_galaxy=source_galaxy_super_compact
)

"""
__Fitting__

we'll make one more convenience function which fits the simulated imaging data with an `Overlay` image-mesh, 
`Delaunay` mesh  and `Constant` regularization scheme pixelization.

We'll input the `coefficient` of each fit, so that for each simulated source we regularize it at an appropriate level. 
There is nothing new in this function you haven't seen before.
"""


def fit_with_delaunay_from(dataset, mask, coefficient):
    dataset = dataset.apply_mask(mask=mask)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.Isothermal(
            centre=(0.0, 0.0), ell_comps=(0.111111, 0.0), einstein_radius=1.6
        ),
    )

    pixelization = al.Pixelization(
        image_mesh=al.image_mesh.Overlay(shape=(30, 30)),
        mesh=al.mesh.Delaunay(),
        regularization=al.reg.Constant(coefficient=coefficient),
    )

    source_galaxy = al.Galaxy(redshift=1.0, pixelization=pixelization)

    tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

    return al.FitImaging(dataset=dataset, tracer=tracer)


"""
__Fit Problems__

Lets fit our first source which was simulated using the flattest light profile. One should note that this uses the 
highest regularization coefficient of our 3 fits (as determined by maximizing the Bayesian log evidence).
"""
fit_flat = fit_with_delaunay_from(
    dataset=dataset_source_flat, mask=mask, coefficient=9.2
)

include = aplt.Include2D(mapper_image_plane_mesh_grid=True, mask=True)

fit_plotter = aplt.FitImagingPlotter(fit=fit_flat, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)


print(fit_flat.log_evidence)

"""
The fit was *excellent*. There were effectively no residuals in the fit, and the source has been reconstructed using 
lots of pixels! Nice!

Now, lets fit the next source, which is more compact.
"""
fit_compact = fit_with_delaunay_from(
    dataset=dataset_source_compact, mask=mask, coefficient=3.3
)

fit_plotter = aplt.FitImagingPlotter(fit=fit_compact, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

print(fit_compact.log_evidence)

"""
The fit does not look so good! 

It reconstructs most of the lensed source's structure, but there are two  clear blobs in the residual map where 
the fit is failing to reconstruct the central regions of the source galaxy.

Take a second to think about why this might be. Is it the mesh or how the regularization is applying smoothing?

Finally, lets fit the very compact source. Given that the results for the compact source did not look good, you are 
right in thinking this is going to make things even worse. Again, think about why this might be.
"""
fit_super_compact = fit_with_delaunay_from(
    dataset=dataset_source_super_compact, mask=mask, coefficient=3.1
)

fit_plotter = aplt.FitImagingPlotter(fit=fit_super_compact, include_2d=include)
fit_plotter.subplot_fit()
fit_plotter.subplot_of_planes(plane_index=1)

print(fit_super_compact.log_evidence)

"""
__Discussion__

Okay, so what did we learn? The more compact our source, the worse the fit. This happens even though we are using the 
correct lens mass model, telling us that something is going fundamentally wrong with our source reconstruction and 
pixelization. Both the mesh and regularization are to blame!

*Image-Mesh / Mesh*:

The problem is the same one we discussed when we compared the `Rectangular` and `Delaunay` meshes in tutorial 7. 

We are simply not dedicating enough source-pixels to the central regions of the source reconstruction, 
e.g. where it`s brightest. As the source becomes more compact, the source reconstruction no longer has enough 
resolution to resolve its fine-detailed central structure, causing the fit to the image to degrade.

As we made our sources more compact we go from reconstructing them using ~100 source pixels, to ~20  source pixels 
to ~ 10 source pixels. This is why we advocated not using the `Rectangular` mesh previously!

Adapting to the mass model magnification is not the best approach. As we simulated more compact sources the 
magnification (which is determined via the mass model) does not change. We therefore reconstructed each source
using fewer and fewer pixels, leading to a worse and worse fit. 

Furthermore, the source galaxies above are located in the highest magnification regions of the source plane! If the 
source's were further away from the caustic, the pixelization would use *even less* pixels to reconstruct them. 
Clearly, we want to adapt to something else. 

**Regularization**:

Regularization also causes problems. When using a `ConstantSplit` regularization scheme, we regularize the source by 
adding up the difference in fluxes between all source-pixels multiplied by one single value, the regularization
coefficient. This means that, every single source pixel receives the same `level` of regularization, regardless of 
whether it is reconstructing the bright central regions of the source or its faint exterior regions. 

To visualize this, we are going to plot the `regularization_weights`. 

The `FitImagingPlotter` does not have a method that is able to plot this attribute of the `Inversion`. However, 
the `FitImagingPlotter` has its own  `InversionPlotter` which we can use to make this plot. The benefit of using this 
is that it inherits from the `FitImagingPlotter` properties like the caustics, so they appear on the figure (this 
would not happen if we manually set up an `InversionPlotter` as we did in previous tutorials.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d_of_pixelization(
    pixelization_index=0, regularization_weights=True
)

"""
As you can see, all pixels are regularized with our input regularization_coefficient value of 3.6.

This is not the best approach to regularizing the source. In fact, different regions of the source prefer different 
levels of regularization:

 1) In the source's central regions its flux gradient is steepest; the change in flux between two source pixels is 
 much larger than in the exterior regions where the gradient is flatter (or there is no source flux at all). To 
 reconstruct the detailed structure of the source's cuspy inner regions, the regularization coefficient needs to 
 be much lower to avoid over-smoothing.

 2) On the other hand, the source reconstruction wants to assume a high regularization coefficient further out 
 because the source's flux gradient is flat (or there is no source signal at all). Higher regularization coefficients 
 will increase the Bayesian evidence because by smoothing more source-pixels it makes the solution `simpler`, given 
 that correlating the flux in these source pixels the solution effectively uses fewer source-pixels (e.g. degrees of 
 freedom).

This outlines the problem using a constant regularization scheme. Some parts of the reconstructed source demand a 
low regularization coefficient whereas other parts want a high value. 

By using a single regularization coefficient, we infer an intermediate regularization coefficient that over-smooths 
the source's central regions whilst failing to fully correlate exterior pixels. 

An adaptive regularization scheme, where the regularization coefficient varies from the outskirts to the centrel 
regions, will produce solutions that further increase the Bayesian evidence.

__Wrap Up__

We have motivated an even more adaptive pixelization, where the mesh and regularization scheme adapt to the source's 
unlensed morphology. 

The next two tutorials will show **PyAutoLens**'s adapt-model which enables this.
"""
