"""
Modeling Features: One By One
=============================

Multi-wavelength analysis does not necessarily require us to fit all datasets simultaneously. Instead, we can fit one
dataset first in order to infer a robust lens and source model, and then fit the next dataset, using the inferred
model as the starting point.

There are many occasions where this approach is beneficial, for example:

- When certain datasets are worse quality (e.g. lower resolution) than others. Fitting them simultaneously may mean this
  dataset's lower quality makes the model fit less robust. By fitting them one by one, using the inferred model of the
  best dataset first, we can ensure the model-fit is as robust as possible and interpret the results of the lower
  quality datasets more clearly.

- It can often produce faster run times, as although more non-linear searches are performed, each search is faster
  than a search which fits all datasets simultaneously.

- To investigate whether lens modeling results inferred when we model all datasets simultanoeusly are robust. If the
  result disappears for fits to individual datasets, this may suggest the result is not robust.

To perform modeling one-by-one, we have to make decision about how simple or complex we make the model after
fitting the highest quality dataset. For example, we may:

- Fix the lens mass model and only allow the lens light and source light to vary.

- Fix the lens mass model and the majority of lens light and source light parameters, allowing only the `intensity`
  values to vary.

- Allow all parameters to vary, but use the highest quality dataset's inferred model as the starting point.

- Whether to account for offsets between the datasets, or to assume the datasets are aligned.

We illustrate different examples in this script, with the appropriate choice depending on your specific science case.

__Model__

This script fits an `Imaging` dataset of a 'galaxy-scale' strong lens with a model where:

 - The lens galaxy's light is a linear parametric linear `Sersic` bulge.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is a linear parametric linear `Sersic`.

__Start Here Notebook__

If any code in this script is unclear, refer to the `modeling/start_here.ipynb` notebook.
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
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band) and red (r-band).

The strings are used for load each dataset.
"""
color_list = ["g", "r"]

"""
__Pixel Scales__

Every multi-wavelength dataset can have its own unique pixel-scale.
"""
pixel_scales_list = [0.08, 0.12]

"""
__Dataset__

Load and plot each multi-wavelength strong lens dataset, using a list of their waveband colors.

The plotted images show that the datasets have a small offset between them, half a pixel based on the resolution of
the second image.
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "dataset_offsets"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset_list = [
    al.Imaging.from_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        pixel_scales=pixel_scales,
    )
    for color, pixel_scales in zip(color_list, pixel_scales_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Mask__

Define a 3.0" circular mask, which includes the emission of the lens and source galaxies.

For multi-wavelength lens modeling, we use the same mask for every dataset whenever possible. This is not
absolutely necessary, but provides a more reliable analysis.
"""
mask_list = [
    al.Mask2D.circular(
        shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
    )
    for dataset in dataset_list
]

dataset_list = [
    dataset.apply_mask(mask=mask) for imaging, mask in zip(dataset_list, mask_list)
]

for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Analysis__

We create an `Analysis` object for every dataset.

We do not sum the analyses, like we do in most other example scripts, as we are going to fit each dataset one-by-one.
"""
analysis_list = [al.AnalysisImaging(dataset=dataset) for dataset in dataset_list]

"""
__Model__

We compose a lens model where:

 - The lens galaxy's light is a linear parametric `Sersic`, where the `intensity` parameter of the lens galaxy
   is solved for linearly [6 parameters].

 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear` [7 parameters].
 
 - The source galaxy's light is a linear parametric `Sersic`, where the `intensity` parameter of the lens galaxy
   is solved for linearly [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=19.
"""
lens = af.Model(
    al.Galaxy,
    redshift=0.5,
    bulge=al.lp_linear.Sersic,
    mass=al.mp.Isothermal,
    shear=al.mp.ExternalShear,
)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp_linear.SersicCore)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__
"""
search = af.Nautilus(
    path_prefix=path.join("multi", "modeling"),
    name="one_by_one__main_dataset",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=4,
)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis_list[0])

"""
__Result__

The result object returned by this model-fit is a `Result` object. It is not a list like other examples, because we 
did not use a combined analysis.
"""
print(result.max_log_likelihood_instance)

"""
Plotting the result's tracer shows the source,
"""
tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.uniform
)
tracer_plotter.subplot_tracer()

fit_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()

"""
__Second Dataset Mass Model Fixed__

We now fit the second dataset using the inferred model of the first dataset as the starting point.

We compose a simple lens model where the mass model is fixed to the result of the first dataset fit, and the lens
and source galaxy's light are varied. 

This model therefore assumes that the mass does not change over wavelength, but the lens and source light do, which
is what we expect for a strong lens system.

The code below uses the search chaining API to link the priors between model parameters, if you are not
familiar with this feature, checkout the `imaging/advanced/chaining` package.
"""
# model = af.Collection(
#     galaxies=af.Collection(
#         lens=af.Model(
#             al.Galaxy,
#             redshift=result.instance.galaxies.lens.redshift,
#             bulge=result.model.galaxies.lens.bulge,
#             mass=result.instance.galaxies.lens.mass,
#             shear=result.instance.galaxies.lens.shear,
#         ),
#         source=result.model.galaxies.source,
#     ),
# )
#
# print(model.info)
#
# search = af.Nautilus(
#     path_prefix=path.join("multi", "modeling"),
#     name="one_by_one__second_mass_model_fixed",
#     unique_tag=dataset_name,
#     n_live=100,
#     number_of_cores=4,
# )
#
# result_mass_model_fixed = search.fit(model=model, analysis=analysis_list[0])

"""
__Second Dataset Offset__

Multi-wavelength datasets often have offsets between their images, which are due to the different telescope pointings
during the observations.

These offsets are often accounted for during the data reduction process, but may not be perfectly corrected and
have uncertainties associated with them.

Fitting datasets one-by-one offers a straightforward method to account for these offsets, by allowing the offset
between the datasets to vary during the model-fit as two free parameters (y and x).

We now fit for the offset between datasets, keeping all lens model parameters fixed to the result of the first dataset
fit. 

In this example, the two datasets are not offset, so the model-fit will infer an offset consistent with (0.0", 0.0").
"""
dataset_model = af.Model(al.DatasetModel)

dataset_model.grid_offset.grid_offset_0 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)
dataset_model.grid_offset.grid_offset_1 = af.UniformPrior(
    lower_limit=-0.1, upper_limit=0.1
)

model = af.Collection(
    dataset_model=dataset_model,
    galaxies=result.instance.galaxies,
)

print(model.info)

search = af.Nautilus(
    path_prefix=path.join("multi", "modeling"),
    name="one_by_one__dataset_offset",
    unique_tag=dataset_name,
    n_live=100,
    number_of_cores=4,
)

result = search.fit(model=model, analysis=analysis_list[0])

"""
Checkout `autolens_workspace/*/imaging/results` for a full description of analysing results in **PyAutoLens**.
"""
