"""
Example: Known Deflections Model
================================

This example is a continuation of the script `input_deflections_sourre_planes.py`. You should read through that
script if you have not done so already before covering this script.

As we discussed, we can now use an input deflection angle map from an external source to create lensed images of
source galaxies. In this example, we assume the source is not known and something we fit for via lens modeling.
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
To begin, we set up the `InputDeflections` object in an identical fashion to the previous example.

In this example, our `input` deflection angle map is the true deflection angles of the `Imaging` data simulated in the 
`mass_sie__source_lp.py` simulator. You should be able to simply edit the `from_fits` methods below to point
to your own dataset an deflection maps.

Load and plot this dataset.
"""
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()

"""
In `autolens_workspace/examples/misc/files` you`ll find the script `make_source_plane.py`, which creates the image-plane 
`Grid2D` and deflection angles we use in this example (which are identical to those used in the 
`mass_sie__source_lp.py` simulator). 

Load the input deflection angle map from a .fits files (which is created in the code mentioned above).
"""
deflections_y = al.Array2D.from_fits(
    file_path=path.join("dataset", "misc", "deflections_y.fits"),
    pixel_scales=dataset.pixel_scales,
)
deflections_x = al.Array2D.from_fits(
    file_path=path.join("dataset", "misc", "deflections_x.fits"),
    pixel_scales=dataset.pixel_scales,
)

"""
Lets plot the deflection angles to make sure they look like what we expect!
"""
aplt.Array2DPlotter(array=deflections_y)
aplt.Array2DPlotter(array=deflections_x)

"""
Lets next load and plot the image-plane grid
"""
grid = al.Grid2D.from_fits(
    file_path=path.join("dataset", "misc", "grid.fits"),
    pixel_scales=dataset.pixel_scales,
)
grid_plotter = aplt.Grid2DPlotter(grid=grid)
grid_plotter.figure_2d()

"""
The `Mask2D` our model-fit using the `InputDeflections` will use. This is set up the same way as the previous script, but
not this `Mask2D` now defines the image-plane region we will fit the data (and therefore where our residuals, chi-squared,
likelihood, etc is calculated.
"""
mask = al.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

grid = al.Grid2D.from_mask(mask=mask)

dataset = dataset.apply_mask(mask=mask)

"""
We create the `InputDeflections` `MassProfile`.almosst the same as the previous example. This is going to be passed to 
a  `Model` below, so we can use it with a source model to fit to the `Imaging` data using a non-linear search.

However, we pass two additional parameters, `preload_grid` and `preload_blurring_grid`. 

The interpolation performed by the `InputDeflections` can be computationally slow, and if we did it for every 
lens model we fit to the data we`d waste a lot of time. However, because our deflection angle map is fixed and the 
`grid` and `blurring_grid` we interpolated it to are fixed, by passing the latter as a `preload_grid` we can skip
this expensive repeated calculation and speed up the code significantly.
"""
image_plane_grid = al.Grid2D.uniform(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)

input_deflections = al.mp.InputDeflections(
    deflections_y=deflections_y,
    deflections_x=deflections_x,
    image_plane_grid=image_plane_grid,
    preload_grid=grid,
    preload_blurring_grid=al.Grid2D.blurring_grid_from(
        mask=mask, kernel_shape_native=dataset.psf.shape_native
    ),
)

"""
__Model__

We now compose the lens and source `Model`, where the source is an `Sersic`.
"""
lens = af.Model(al.Galaxy, redshift=0.5, mass=input_deflections)
source = af.Model(al.Galaxy, redshift=1.0, bulge=al.lp.Sersic)

model = af.Collection(galaxies=af.Collection(lens=lens, source=source))

"""
__Search__

The source is fitted to the `Imaging` data via the input deflection angles using a non-linear search, which we 
specify below as the nested sampling algorithm Nautilus (https://nautilus-sampler.readthedocs.io/en/latest/). Checkout 
other examples on the workspace if you are unsure what this does!

The script `autolens_workspace/*/modeling/customize/non_linear_searches.py` gives a description of the types of
non-linear searches that **PyAutoLens** supports. If you do not know what a non-linear search is or how it 
operates, checkout chapters 1 and 2 of the HowToLens lecture series.
"""
search = af.Nautilus(
    path_prefix=path.join("misc"),
    name="search__input_deflections",
    unique_tag=dataset_name,
    n_live=150,
)

"""
__Analysis__

Create the `AnalysisImaging` object defining how the via Nautilus the model is fitted to the data.
"""
analysis = al.AnalysisImaging(dataset=dataset)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the folder `autolens_workspace/output/imaging/simple__no_lens_light/mass[sie]_source[bulge]` for live outputs 
of the results of the fit, including on-the-fly visualization of the best fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
