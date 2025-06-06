"""
Modeling: Quantity
==================

Ordinary model-fits in **PyAutoLens** fit a lens model to a dataset (e.g. `Imaging`, `Interferometer`). The inferred
lens model then tells us about the properties of the lens galaxy, for example its convergence, potential and
deflection angles.

This script instead fits a lens model directly to a quantity of lens galaxy, which could be its convergence,
potential, deflection angles or another of its quantities.

This fit allows us to fit a quantity of a certain mass profile (e.g. the convergence of an `NFW` mass profile) to
the same quantity of a different mass profile (e.g. the convergence of a `PowerLaw`). This provides parameters
describing how to translate between two mass profiles as closely as possible, and to understand how similar or
different the mass profiles are.

This script fits a `DatasetQuantity` dataset of a 'galaxy-scale' strong lens with a model. The `DatasetQuantity` is the
convergence map of a `NFW` mass model which is fitted by a lens model where:

 - The lens galaxy's light is omitted (and is not present in the simulated data).
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - There is no source galaxy.
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
__Grid__

Define the 2D grid the quantity (in this example, the convergence) is evaluated using.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

"""
__Tracer__

Create a tracer which we will use to create our `DatasetQuantity`. 
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.NFW(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        kappa_s=0.2,
        scale_radius=20.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy])

"""
__Dataset__

Use this `Tracer`'s 2D convergence to create the `DatasetQuantity`.

We assume a noise-map where all values are arbritrarily 0.01.
"""
convergence_2d = tracer.convergence_2d_from(grid=grid)

dataset = al.DatasetQuantity(
    data=convergence_2d,
    noise_map=al.Array2D.full(
        fill_value=0.01,
        shape_native=convergence_2d.shape_native,
        pixel_scales=convergence_2d.pixel_scales,
    ),
)

"""
__Mask__

The model-fit requires a `Mask2D` defining the regions of the convergence we fit, which we define and apply to the 
`DatasetQuantity` object.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask)

"""
__Model__

We compose a lens model where:

 - The lens galaxy's total mass distribution is an `PowerLaw` [6 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=14.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=al.mp.PowerLaw)

model = af.Collection(galaxies=af.Collection(lens=lens))

"""
__Search__

The lens model is fitted to the data using the nested sampling algorithm Nautilus (see `start.here.py` for a 
full description).

The folders: 

 - `autolens_workspace/*/modeling/imaging/searches`.
 - `autolens_workspace/*/modeling/imaging/customize`
  
Give overviews of the non-linear searches **PyAutoLens** supports and more details on how to customize the
model-fit, including the priors on the model.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autolens_workspace/output/modeling/imaging/simple__no_lens_light/mass[sie]_source[bulge]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.

An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Nautilus uses parallel processing to sample multiple 
lens models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.
"""
search = af.Nautilus(
    path_prefix=path.join("misc", "modeling"),
    name="quantity_via_convergence_fit",
    n_live=100,
    number_of_cores=1,
)

"""
__Analysis__

The `AnalysisQuantity` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `DatasetQuantity` dataset.

This includes a `func_str` input which defines what quantity is fitted. It corresponds to the function of the 
model `Tracer` objects that are called to create the model quantity. For example, if `func_str="convergence_2d_from"`, 
the convergence is computed from each model `Tracer`.
"""
analysis = al.AnalysisQuantity(dataset=dataset, func_str="convergence_2d_from")

"""
__Model-Fit__

We begin the model-fit by passing the model and analysis object to the non-linear search (checkout the output folder
for on-the-fly visualization and results).
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which whose `info` attribute shows the result in a readable format:
"""
print(result.info)

"""
We plot the maximum likelihood fit, tracer images and posteriors inferred via Nautilus.

Checkout `autolens_workspace/*/results` for a full description of analysing results in **PyAutoLens**.
"""
print(result.max_log_likelihood_instance)

tracer_plotter = aplt.TracerPlotter(
    tracer=result.max_log_likelihood_tracer, grid=result.grids.lp
)
tracer_plotter.subplot_tracer()

fit_quantity_plotter = aplt.FitQuantityPlotter(fit=result.max_log_likelihood_fit)
fit_quantity_plotter.subplot_fit()

plotter = aplt.NestPlotter(samples=result.samples)
plotter.corner_anesthetic()

"""
Checkout `autolens_workspace/*/modeling/imaging/results.py` for a full description of the result object.
"""
