"""
Pipelines: Light Parametric + Mass Total + Source Inversion
===========================================================

By chaining together five searches this script  fits `Imaging` dataset of a 'galaxy-scale' strong lens, where in the final model:

 - The lens galaxy's light is a parametric `EllSersic` and `EllExponential`.
 - The lens galaxy's total mass distribution is an `EllIsothermal`.
 - The source galaxy is modeled using an `Inversion`.
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

"""
__Dataset__ 

Load the `Imaging` data, define the `Mask2D` and plot them.
"""
dataset_name = "light_sersic_exp__mass_sie__source_sersic"
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
path_prefix = path.join("imaging", "pipelines")

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Model + Search + Analysis + Model-Fit (Search 1)__

In search 1 we fit a lens model where:

 - The lens galaxy's light is a parametric `EllSersic` bulge and `EllExponential` disk, the centres of 
 which are aligned [11 parameters].
 
 - The lens galaxy's mass and source galaxy are omitted.

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
"""
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
    name="search[1]_light[parametric]",
    unique_tag=dataset_name,
    nlive=50,
)

analysis = al.AnalysisImaging(dataset=imaging)

result_1 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 2)__

In search 2 we fit a lens model where:

 - The lens galaxy's light is an `EllSersic` bulge and `EllExponential` disk [Parameters fixed to results 
 of search 1].
 
 - The lens galaxy's total mass distribution is an `EllIsothermal` with `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a parametric `EllSersic` [7 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
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
    name="search[2]_light[fixed]_mass[sie]_source[parametric]",
    unique_tag=dataset_name,
    nlive=75,
)

analysis = al.AnalysisImaging(dataset=imaging)

result_2 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 3)__

In search 2 we fit a lens model where:

 - The lens galaxy's light is an `EllSersic` bulge and `EllExponential` disk [11 Parameters: we do not
  use the results of search 1 to initialize priors].
  
 - The lens galaxy's total mass distribution is an `EllIsothermal` with `ExternalShear` [7 parameters: priors
 initalized from search 2].
 
 - The source galaxy's light is a parametric `EllSersic` [7 parameters: priors initalized from search 2].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=25.

NOTES:

 - The result of search 1 is sufficient for subtracting the lens light, so that search 2 can accurately fit the lens
 mass model and source light. However, the lens light model may not be particularly accurate, so we opt not to use
 the result of search 1 to initialize the priors.
"""
bulge = af.Model(al.lp.EllSersic)
disk = af.Model(al.lp.EllExponential)

bulge.centre = disk.centre

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
    name="search[3]_light[parametric]_mass[total]_source[parametric]",
    unique_tag=dataset_name,
    nlive=100,
)

analysis = al.AnalysisImaging(dataset=imaging)

result_3 = search.fit(model=model, analysis=analysis)

"""
__Model + Search + Analysis + Model-Fit (Search 4)__

We use the results of searches 3 to create the lens model fitted in search 4, where:

 - The lens galaxy's light is an `EllSersic` bulge and `EllExponential` disk [Parameters fixed to 
 results of search 3].
 
 - The lens galaxy's total mass distribution is an `EllIsothermal` and `ExternalShear` [Parameters fixed to 
 results of search 3].
 
 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [2 parameters].
 
 - This pixelization is regularized using a `Constant` scheme [1 parameter]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=3.

NOTES:

 - This search allows us to very efficiently set up the resolution of the pixelization and regularization coefficient 
 of the regularization scheme, before using these models to refit the lens mass model.
"""
model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_3.instance.galaxies.lens.bulge,
            disk=result_3.instance.galaxies.lens.disk,
            mass=result_3.instance.galaxies.lens.mass,
            shear=result_3.instance.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=al.pix.VoronoiMagnification,
            regularization=al.reg.Constant,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[4]_light[fixed]_mass[fixed]_source[inversion_initialization]",
    unique_tag=dataset_name,
    nlive=20,
)

analysis = al.AnalysisImaging(dataset=imaging)

result_4 = search.fit(model=model, analysis=analysis)

"""
__Model +  Search (Search 5)__

We use the results of searches 3 and 4 to create the lens model fitted in search 5, where:

 - The lens galaxy's light is an `EllSersic` bulge and `EllExponential` disk [11 parameters: priors 
 initialized from search 3].
 
 - The lens galaxy's total mass distribution is an `EllPowerLaw` and `ExternalShear` [8 parameters: priors 
 initialized from search 3].
 
 - The source-galaxy's light uses a `VoronoiMagnification` pixelization [parameters fixed to results of search 4].
 
 - This pixelization is regularized using a `Constant` scheme [parameters fixed to results of search 4]. 

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.
"""
mass = af.Model(al.mp.EllPowerLaw)
mass.take_attributes(result_3.model.galaxies.lens.mass)

model = af.Collection(
    galaxies=af.Collection(
        lens=af.Model(
            al.Galaxy,
            redshift=redshift_lens,
            bulge=result_3.model.galaxies.lens.bulge,
            disk=result_3.model.galaxies.lens.disk,
            mass=mass,
            shear=result_3.model.galaxies.lens.shear,
        ),
        source=af.Model(
            al.Galaxy,
            redshift=redshift_source,
            pixelization=result_4.instance.galaxies.source.pixelization,
            regularization=result_4.instance.galaxies.source.regularization,
        ),
    )
)

search = af.DynestyStatic(
    path_prefix=path_prefix,
    name="search[5]_light[parametric]_mass[total]_source[inversion]",
    unique_tag=dataset_name,
    nlive=50,
)

"""
__Positions + Analysis + Model-Fit (Search 5)__

We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_inversion.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
settings_lens = al.SettingsLens(
    positions_threshold=result_4.positions_threshold_from(
        factor=3.0, minimum_threshold=0.2
    )
)

analysis = al.AnalysisImaging(
    dataset=imaging,
    positions=result_4.image_plane_multiple_image_positions,
    settings_lens=settings_lens,
)

result_5 = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
