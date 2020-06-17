# %%
"""
__Adaptive Pixelization__

In this tutorial we'll use a new pixelization, called the VoronoiMagnification pixelization. This pixelization
 doesn't use a uniform _Grid_ of rectangular pixels, but instead uses an irregular 'Voronoi' pixels. So, why do we
 want to do that? Lets take another quick look at the rectangular grid..
"""

# %%
#%matplotlib inline

import autolens as al
import autolens.plot as aplt

# %%
"""
This simulates the same the image we've fitted in the past few tutorials.
"""

# %%
def simulate():

    grid = al.Grid.uniform(shape_2d=(150, 150), pixel_scales=0.05, sub_size=2)

    psf = al.Kernel.from_gaussian(shape_2d=(11, 11), sigma=0.05, pixel_scales=0.05)

    lens_galaxy = al.Galaxy(
        redshift=0.5,
        mass=al.mp.EllipticalIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.6
        ),
    )

    source_galaxy = al.Galaxy(
        redshift=1.0,
        light=al.lp.EllipticalSersic(
            centre=(0.0, 0.0),
            elliptical_comps=(0.2, 0.1),
            intensity=0.2,
            effective_radius=0.2,
            sersic_index=2.5,
        ),
    )

    tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

    simulator = al.SimulatorImaging(
        exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        psf=psf,
        background_sky_map=al.Array.full(fill_value=100.0, shape_2d=grid.shape_2d),
        add_noise=True,
    )

    return simulator.from_tracer_and_grid(tracer=tracer, grid=grid)


# %%
"""
Lets quickly remind ourselves of the image and the 3.0" circular mask we'll use to mask it.
"""

# %%
imaging = simulate()

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=2, radius=2.5
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
The lines of code below do everything we're used to, that is, setup an image, mask it, trace it via a tracer, 
setup the rectangular mapper, etc.
"""

# %%
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.6
    ),
)

masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.Rectangular(shape=(40, 40)),
    regularization=al.reg.Constant(coefficient=0.5),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

aplt.FitImaging.subplot_fit_imaging(fit=fit, include=aplt.Include(mask=True))

# %%
"""
Okay, so lets think about the rectangular pixelization. Is this the optimal way to reconstruct our source? Are there 
features in the source-plane that arn't ideal? How do you think we could do a better job?

Well, given we're doing a whole tutorial on using a different pixelization to the rectangular grid, you've probably 
guessed that it isn't optimal. Infact, its pretty rubbish, and not a pixelization we should actually want to model 
many lenses with!

So what is wrong with the grid? Well, lets think about the source reconstruction.
"""

# %%
aplt.Inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

# %%
"""
There is one clear problem, we are using a small number of the total source pixels to reconstruct the source. The 
majority of source pixels are located away from the source. By my estimate, we're using just 16 pixels (the central 
4x4 grid) out of the 1600 pixels to actually fit the data! The remaining ~1500 pixels are doing *nothing* but fit noise.

This is a waste and our analysis will take longer to run because of it. However, more importantly, it means that our 
Bayesian regularization scheme is sub-optimal. In tutorial 4, we discussed how the Bayesian log evidence of the 
regularization wants to obtain the *simplest* source solution possible. That is the solution which fits the data 
well using the fewest source pixels. Clearly, if we dedicating a huge number of source pixels to doing *nothing*, our 
source reconstruction will be unecessarily complex (and therefore lower log_evidence).

If our pixelization could 'focus' its pixels where we actually have more data, e.g. the highly magnified regions of 
the source-plane, we could reconstruct the source using far fewer pixels. That'd be great both for computational 
efficiency and increasing the Bayesian log evidence and that is exactly what our Voronoi _Grid_ does.

To achieve this, we first compute an 'image-plane sparse grid', which is a set of sparse coordinates in the image-plane 
that will be ray-traced to the source-plane and define the centres of our source-pixel grid. We compute this _Grid_ 
directly from a pixelization, by passing it a grid.
"""

# %%
adaptive = al.pix.VoronoiMagnification(shape=(20, 20))

image_plane_sparse_grid = adaptive.sparse_grid_from_grid(grid=masked_imaging.grid)

# %%
"""
We can plot this _Grid_ over the image, to see that it is a coarse _Grid_ over-laying the image itself.
"""

# %%
aplt.Imaging.image(imaging=imaging, grid=image_plane_sparse_grid, mask=mask)

# %%
"""
When we pass a tracer a source galaxy with this pixelization it automatically computes the ray-traced source-plane 
Voronoi _Grid_ using the _Grid_ above. Thus, our Voronoi pixelization is used by the tracer's fit.
"""

# %%
source_galaxy = al.Galaxy(
    redshift=1.0, pixelization=adaptive, regularization=al.reg.Constant(coefficient=1.0)
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

# %%
"""
If we look at the lens fit, we'll that our source-plane no longer uses rectangular pixels, but Voronoi pixels!
"""

# %%
fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)

aplt.FitImaging.subplot_fit_imaging(
    fit=fit,
    include=aplt.Include(
        mask=True,
        inversion_image_pixelization_grid=True,
        inversion_pixelization_grid=True,
    ),
)

# %%
"""
And we can take a closer inspection of the inversion itself.
"""

# %%
aplt.Inversion.reconstruction(
    inversion=fit.inversion, include=aplt.Include(inversion_pixelization_grid=True)
)

# %%
"""
Clearly, this is an improvement. We're using fewer pixels than the rectangular _Grid_ (400, instead of 1600), but 
reconstructing our source is far greater detail. A win all around? It sure is.

On our rectangular grid, we regularized each source pixel with its 4 neighbors. We compared their fluxes, summed 
the differences, and penalized solutions where the differences were large. For a Voronoi grid, we do the same thing, 
now comparing each source-pixel with all other source-pixels with which it shares a direct vertex. This means that 
different source-pixels may be regularized with different numbers of source-pixels, depending on how many neighbors 
are formed.

This Voronoi magnification _Grid_ is still far from optimal. There are lots of source-pixels effectively fitting just 
noise. We may achieve even better solutions if the central regions of the source were reconstructed using even more 
pixels. So, how do we improve on this? Well, you'll have to wait until chapter 5, when we introduce PyAutoLens's 
adaptive functionality, or 'hyper-mode'.

In the mean time, you may wish to experiment with using both Rectangular and VoronoiMagnification _Grid_'s to fit 
lenses which can be easily achieve by changing the input pixeliation given to a pipeline.
"""
