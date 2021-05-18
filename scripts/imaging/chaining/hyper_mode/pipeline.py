"""
Chaining: Hyper-Model Example Pipeline
======================================

Non-linear search chaining is an advanced model-fitting approach in **PyAutoLens** which breaks the model-fitting
procedure down into multiple non-linear searches, using the results of the initial searches to initialization parameter
sampling in subsequent searches. This contrasts the `modeling` examples which each compose and fit a single lens
model-fit using one non-linear search.

An overview of search chaining is provided in the `autolens_workspace/notebooks/imaging/chaining/api.py` script, make
sure to read that before reading this script!

The script `hyper_mode.py` introduces **PyAutoLens**'s hyper-mode, which passes the the results of previous model-fits
performed by earlier searches to searches performed later in the chain. This script gives an example pipeline using
hyper-mode. It is an adaption of the pipeline `chaining/pipelines/light_parametric__mass_total__source_inversion.py`
and it can be used as a template for setting up any pipeline to use hyper-mode.

Hyper mode is also built into the SLaM pipelines by default.

By chaining together five searches this script fits strong lens `Imaging`, where in the final model:

 - The lens galaxy's light is a parametric `EllSersic` and `EllExponential`.
 - The lens galaxy's total mass distribution is an `EllIsothermal`.
 - The source galaxy is modeled using an `Inversion`, in particular the `VoronoiBrightnessImage` pixelization and
 `AdaptiveBrightness` regularization schemes which require hyper-mode.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt
import extensions

"""
__Dataset__ 

Load the `Imaging` data, define the `Mask2D` and plot them.
"""
dataset_name = "light_sersic_exp__mass_sie__source_sersic_x2"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining", "hyper_pipeline")

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__HYPER SETUP__

The `SetupHyper` determines which hyper-mode features are used during the model-fit. The following options are 
available:

 - `hyper_galaxies`: whether the lens and / or source galaxy are treated as a hyper-galaxy, meaning that the model-fit
 can increase the noise-map values in the regions of the lens or source if they are poorly fitted.

 - `hyper_image_sky`: The background sky subtraction may be included in the model-fitting.

 - `hyper_background_noise`: The background noise-level may be included in the model-fitting.

The pixelization and regularization schemes which use hyper-mode to adapt to the source's properties are not passed 
into `SetupHyper`, but are used in this example script below.

In this example, we a hyper galaxy for the lens and include the background sky subtraction in the model.
"""
setup_hyper = al.SetupHyper(
    hyper_galaxies_lens=True,
    hyper_galaxies_source=False,
    hyper_image_sky=al.hyper_data.HyperImageSky,
    hyper_background_noise=None,
)

"""
__Model-Fits via Searches 1, 2 & 3__

Searches 1, 2 and 3 initialize the lens model by fitting the lens light, then the lens mass + source, and then all
simultaneously. This is identical to the pipeline `chaining/pipelines/light_parametric__mass_total__source_inversion.py`

We can only use hyper-model once we have a good model for the lens and source galaxies, given that it needs hyper-model
images of both of these components to effectively perform tasks like scaling their noise or adapting a pixelization
or regularization pattern to the source's unlensed morphology.
"""
analysis = al.AnalysisImaging(dataset=imaging)

bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)

bulge.centre = disk.centre

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(al.Galaxy, redshift=0.5, bulge=bulge, disk=disk)
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[1]_light[parametric]",
    unique_tag=dataset_name,
    nlive=50,
)

result_1 = search.fit(model=model, analysis=analysis)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_1.instance.galaxies.lens.bulge,
            disk=result_1.instance.galaxies.lens.disk,
            mass=al.mp.EllIsothermal,
            shear=al.mp.ExternalShear,
        ),
        source=af.Model(al.Galaxy, redshift=redshift_source, bulge=al.lp.EllSersic),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[2]_light[fixed]_mass[sie]_source[parametric]",
    unique_tag=dataset_name,
    nlive=75,
)

result_2 = search.fit(model=model, analysis=analysis)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=bulge,
            disk=disk,
            mass=result_2.model.galaxies.lens.mass,
            shear=result_2.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            bulge=result_2.model.galaxies.source.bulge,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[3]_light[parametric]_mass[total]_source[parametric]",
    unique_tag=dataset_name,
    nlive=100,
)

analysis = al.AnalysisImaging(dataset=imaging)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Hyper Extension__

Now that Searches 1-3 have provided us with hyper images for the lens and source, we can perform our first use of hyper 
mode, which in this example sees us include the background sky in the model as well a hyper galaxy for the lens which 
scale the noise in the data. 

To activate the hyper-model we extend the above search with a hyper-search. The hyper extension fixes all of the 
non-hyper lens model parameters (e.g. the lens light parameters, mass parameters and source light parameters) and 
fits for only hyper parameters (e.g. the hyper-data components, `Inversion` parameters if included, etc.). 

It therefore depends on the `SetupHyper` object as follows:

 - If the source is using an `Inversion` (does not depend on `SetupHyper`).
 - One or more `HyperGalaxy`'s are included (e.g. if `hyper_galaxies_source` and / or `hyper_galaxies_lens` are True).
 - The background sky is included (if `hyper_image_sky=al.hyper_data.HyperImageSky`).
 - The background noise is included (if `hyper_background_noise=al.hyper_data.HyperBackgroundNoise`)..
 
The hyper extension automatically uses the maximum likelihood model of search 3 to set up the hyper-images.
 
An extension adds an additional result to the result output by the search. For a hyper-extension `result_3` will now
have an addition result attribute that can be accessed via `result_3.hyper`. This is used below to pass the 
"""
result_3 = extensions.hyper_fit(
    setup_hyper=setup_hyper,
    result=result_3,
    analysis=analysis,
    include_hyper_image_sky=True,
)

"""
__Model-Fits via Searches 4 & 5__

Hyper-mode is now scaling the lens and source noise-maps and fitting for the background sky. We now want an `Inversion`
which adapts the pixelization and regularization to the source's morphology. However, our hyper-model images are not
yet sufficently accurate to do this. 

This is because there are two distinct components of the source in the source plane, which the single `EllSersic`
fit above will have failed to capture in detail. If we attempted to use its hyper image to adapt to the source 
morphology, we would only adapt to the single component that we fitted!

We therefore perform two searches which reconstruct the source using an `Inversion`, however this uses a 
`VoronoiMagnification` pixelization and `Constant` regularization, which do not use hyper-model to adapt to the source. 
These will capture both source components ensuring hyper mode is accurate.

This also explains why we set `hyper_galaxies_source=False` in `SetupHyper`, its scaled noise map would have been 
unreliable due to the inaccurate hyper-image. In this example, we will keep the source hyper galaxy turned off, 
but for model-fits where it could be useful it is generally advised that the source hyper galaxy is only switched on
after its hyper-model image is created via an `Inversion`.

You'll note that all hyper-mode examples and the SLaM pipelines use this trick, as using parametric sources to adapt 
to the source morphology can lead to poor results for complex sources.
"""
analysis = al.AnalysisImaging(dataset=imaging, hyper_result=result_3)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_3.instance.galaxies.lens.bulge,
            disk=result_3.instance.galaxies.lens.disk,
            mass=result_3.instance.galaxies.lens.mass,
            shear=result_3.instance.galaxies.lens.shear,
            hyper_galaxy=setup_hyper.hyper_galaxy_lens_from_result(result=result_3),
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.reg.Constant,
        ),
    ),
    hyper_image_sky=result_3.hyper.instance.hyper_image_sky,
    hyper_background_noise=result_3.hyper.instance.hyper_background_noise,
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[4]_light[fixed]_mass[fixed]_source[inversion_initialization]",
    unique_tag=dataset_name,
    nlive=20,
)

result_4 = search.fit(model=model, analysis=analysis)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=result_4.instance.galaxies.lens.redshift,
            bulge=result_4.instance.galaxies.lens.bulge,
            disk=result_4.instance.galaxies.lens.disk,
            mass=result_3.model.galaxies.lens.mass,
            shear=result_3.model.galaxies.lens.shear,
            hyper_galaxy=result_4.instance.galaxies.lens.hyper_galaxy,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=result_4.instance.galaxies.source.redshift,
            pixelization=result_4.instance.galaxies.source.pixelization,
            regularization=result_4.instance.galaxies.source.regularization,
            hyper_galaxy=result_4.instance.galaxies.source.hyper_galaxy,
        ),
    ),
    hyper_image_sky=result_4.instance.hyper_image_sky,
    hyper_background_noise=result_4.instance.hyper_background_noise,
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[5]_light[fixed]_mass[total]_source[inversion_magnification]",
    unique_tag=dataset_name,
    nlive=50,
)

result_5 = search.fit(model=model, analysis=analysis)

"""
__Model-Fits via Searches 6 & 7__

We are now ready to use hyper-model to adapt the `Inversion` to the source's unlensed morphology, given that the 
model-fit above will give us reliable hyper images.

__Preloads__: 
 
Calculating the source-plane pixel grid of a `VoronoiBrightnessImage` pixelization is computationally expensive, slowing
down the time a log likelihood evaluation takes in **PyAutoLens**. However, when the source pixelization is fixed there
is no need recalculate the centre of every source-pixel centre (in the image-plane). 

This is the case in search 7 and we can therefore set up the analysis with a `Preload` object containing the image 
pixel source pixel centres, which will then not be recalculated every iteration of the likelihood function. This speeds
up the model-fit significantly.
  
This uses the maximum likelihood hyper-result search 6. 
"""
analysis = al.AnalysisImaging(dataset=imaging, hyper_result=result_5)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[6]_light[fixed]_mass[fixed]_source[inversion_initialization]",
    unique_tag=dataset_name,
    nlive=30,
    dlogz=10.0,
    sample="rstagger",
)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=result_5.instance.galaxies.lens.redshift,
            bulge=result_5.instance.galaxies.lens.bulge,
            disk=result_5.instance.galaxies.lens.disk,
            mass=result_5.instance.galaxies.lens.mass,
            shear=result_5.instance.galaxies.lens.shear,
            hyper_galaxy=result_5.instance.galaxies.lens.hyper_galaxy,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=result_5.instance.galaxies.source.redshift,
            pixelization=al.pix.VoronoiBrightnessImage,
            regularization=al.reg.AdaptiveBrightness,
            hyper_galaxy=result_5.instance.galaxies.source.hyper_galaxy,
        ),
    ),
    hyper_image_sky=result_5.instance.hyper_image_sky,
    hyper_background_noise=result_5.instance.hyper_background_noise,
)

result_6 = search.fit(model=model, analysis=analysis)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=result_6.instance.galaxies.lens.redshift,
            bulge=result_6.instance.galaxies.lens.bulge,
            disk=result_6.instance.galaxies.lens.disk,
            mass=result_5.model.galaxies.lens.mass,
            shear=result_5.model.galaxies.lens.shear,
            hyper_galaxy=result_6.instance.galaxies.lens.hyper_galaxy,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=result_6.instance.galaxies.source.redshift,
            pixelization=result_6.instance.galaxies.source.pixelization,
            regularization=result_6.instance.galaxies.source.regularization,
            hyper_galaxy=result_6.instance.galaxies.source.hyper_galaxy,
        ),
    ),
    hyper_image_sky=result_6.instance.hyper_image_sky,
    hyper_background_noise=result_6.instance.hyper_background_noise,
)

preloads = al.Preloads.setup(result=result_6, pixelization=True)

analysis = al.AnalysisImaging(dataset=imaging, hyper_result=result_5, preloads=preloads)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[7]_light[fixed]_mass[total]_source[inversion]",
    unique_tag=dataset_name,
    nlive=50,
)

result_7 = search.fit(model=model, analysis=analysis)

"""
__Hyper Extension__

We perform another hyper-extension, which updates the hyper-galaxy noise scaling map and background sky model (which 
were fixed throughout searches 4-7) using the new hyper-model images as well as doing this whilst simultaneously
fitting the `Inversion` parameters.

Note how this extension will use the hyper model images computed in search 7, which use the `VoronoiBrightnessImage`
pixelization and `AdaptiveBrightness` regularization and therefore should provide a really accurate hyper image of
the source galaxy.
"""
result_7 = extensions.hyper_fit(
    setup_hyper=setup_hyper,
    result=result_7,
    analysis=analysis,
    include_hyper_image_sky=True,
)

"""
__Model-Fits Search 8__

Searches 1-7 were the steps we had to go through to properly initialize every aspect of the model for hyper-mode.
The most notable challenges were ensuring that our source hyper image could fully account for an irregular source
with multiple components.

The final search in this hyper-pipeline fits an `EllPowerLaw` mass model, which benefits a lot from hyper-mode
as the `slope` is a difficult parameter to infer which relies heavily on the intricacies of how the source is 
reconstructed. 

__Preloads__: 

We again preload this with the image-pixel source pixel centres, now using the hyper-result of search 7.
"""
mass = af.Model(al.mp.EllPowerLaw)
mass.take_attributes(result_7.model.galaxies.lens.mass)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_7.model.galaxies.lens.bulge,
            disk=result_7.model.galaxies.lens.disk,
            mass=mass,
            shear=result_7.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=result_7.hyper.instance.galaxies.source.pixelization,
            regularization=result_7.hyper.instance.galaxies.source.regularization,
        ),
    )
)

preloads = al.Preloads.setup(result=result_7.hyper, pixelization=True)

analysis = al.AnalysisImaging(
    dataset=imaging, hyper_result=result_7.hyper, preloads=preloads
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="hyper[8]_light[parametric]_mass[total]_source[inversion]",
    unique_tag=dataset_name,
    nlive=50,
)

result_8 = search.fit(model=model, analysis=analysis)

"""
__Wrap Up__

It took us 7 searches to set up hyper-mode, just so that we could fit a complex lens model in one final search. However,
this is what is unfortunately what is necessary to fit the most complex lens models accurately, as they really are
trying to extract a signal that is contained in the intricate detailed surfaceness brightness of the source itself.

The final search in this example fitting an `EllPowerLaw`, but it really could have been any of the complex
models that are illustrated throughout the workspace (e.g., decomposed light_dark models, more complex lens light
models, etc.). You may therefore wish to adapt this pipeline to fit the complex model you desire for your science-case,
by simplying swapping out the model used in search 8.
 
However, it may instead be time that you check out the for the SLaM pipelines, which have hyper-mode built in but 
provide a lot more flexibility in customizing the model and fitting procedure to fully exploit the hyper-mode features
whilst fitting many different lens models.
"""
