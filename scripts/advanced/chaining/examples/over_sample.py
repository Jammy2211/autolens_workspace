"""
Chaining: Over Sample
=====================

Over sampling is a numerical technique where the images of light profiles and galaxies are evaluated
on a higher resolution grid than the image data to ensure the calculation is accurate.

If you are reading this example, you should be familiar with over sampling already. If this is not the case,
checkout the over sampling guide at `autolens_workspace/*/guides/over_sampling.py`.

The guide illustrated adaptive over sampling, where the over sampling sub grid used high resolution pixels in the
centre of a light profile and lower resolution pixels further out. This reached high levels of numerical accuracy
with efficient run times.

However, the adaptive sub grid only works for uniform grids which have not been deflected or ray-traced by the lens
mass model. This criteria is met for the lens galaxy's light, but not for the emission of the lensed source. There
is no automatic adaptive method for the lensed source, which is why the autolens workspace uses cored light profiles
throughout.

An efficient and adaptive over sampling grid is possible. However, it requires using search chaining, where between
searches the over sampling grid is updated.

This example shows how to combine lensed source adaptive over sampling with search chaining.

__Start Here Notebook__

If any code in this script is unclear, refer to the `chaining/start_here.ipynb` notebook.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())

"""
__Dataset + Masking__ 

Load, plot and mask the `Imaging` data.

The data is simulated using a Sersic without a core, unlike most datasets fitted throughout the workspace.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
__Paths__

The path the results of all chained searches are output:
"""
path_prefix = path.join("imaging", "chaining")

"""
__Redshifts__

The redshifts of the lens and source galaxies.
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model_1 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_1.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.

Faint residuals around the multiple images will be present, because the simulated data used a non-cored Sersic
whereas the model fitted is a cored Sersic.
"""
search_1 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[1]__sersic_core",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_1 = al.AnalysisImaging(dataset=dataset)

result_1 = search_1.fit(model=model_1, analysis=analysis_1)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_1.info)

"""
__Over Sampling (Search 2)__

We now create an over sampling grid which applies high levels of over sampling to the brightest regions of the
lensed source in the image plane.

This uses the result of the first lens model in the following way:

 1) Use the lens mass model to ray-trace every deflected image pixel to the source plane, computed the traced grid.
  
 2) Use the traced grid and the centre of the source light profile to compute the distance of every traced image pixel 
    to the source centre. 
    
 3) For all pixels with a distance below a threshold value of 0.1", we set the over sampling factor to a high value of 
    32, which will ensure accuracy in the evaluated of the source's light profile, even after lensing. Pixels 0.1" to
    0.3" from the centre use an over sampling factor of 4, and all other pixels use an over sampling factor of 2.
 
 4) Pass this adaptive over sampling grid to the dataset so it is used in the second model-fit.   
    
"""

tracer = result_1.max_log_likelihood_tracer

traced_grid = tracer.traced_grid_2d_list_from(
    grid=dataset.grid,
)[-1]

source_centre = tracer.galaxies[1].bulge.centre

dataset = dataset.apply_over_sampling(
    over_sampling=al.OverSamplingDataset(
        non_uniform=al.OverSamplingUniform.from_radial_bins(
            grid=traced_grid,
            sub_size_list=[32, 8, 2],
            radial_list=[0.1, 0.3],
            centre_list=[source_centre],
        )
    )
)

"""
__Model (Search 1)__

Search 1 fits a lens model where:

 - The lens galaxy's total mass distribution is an `Isothermal` with `ExternalShear` [7 parameters].
 - The source galaxy's light is a linear parametric `SersicCore` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
lens = af.Model(
    al.Galaxy, redshift=0.5, mass=al.mp.Isothermal, shear=al.mp.ExternalShear
)

source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.Sersic)

model_2 = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
The `info` attribute shows the model in a readable format.
"""
print(model_2.info)

"""
__Search + Analysis + Model-Fit (Search 1)__

We now create the non-linear search, analysis and perform the model-fit using this model.

You may wish to inspect the results of the search 1 model-fit to ensure a fast non-linear search has been provided that 
provides a reasonably accurate lens model.

Faint residuals around the multiple images will be present, because the simulated data used a non-cored Sersic
whereas the model fitted is a cored Sersic.
"""
search_2 = af.Nautilus(
    path_prefix=path_prefix,
    name="search[2]__sersic_over_sampled",
    unique_tag=dataset_name,
    n_live=100,
)

analysis_2 = al.AnalysisImaging(dataset=dataset)

result_2 = search_2.fit(model=model_2, analysis=analysis_2)

"""
__Result (Search 1)__

The results which are used for prior passing are summarized in the `info` attribute.
"""
print(result_2.info)

"""
Fin.
"""
