"""
Overview: Groups
----------------

The strong lenses we've discussed so far have just a single lens galaxy responsible for the lensing, with a single
source galaxy observed.

A strong lensing group is a system which has a distinct 'primary' lens galaxy and a handful of lower mass galaxies
nearby. They typically contain just one or two lensed sources whose arcs are extended and visible. Their Einstein
Radii range between typical values of 5.0" -> 10.0" and with care, it is feasible to fit the source's extended
emission in the imaging or interferometer data.

Strong lensing clusters, which contain many hundreds of lens and source galaxies, are covered in the next overview.
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

In this overview lets begin on the group scale with a simulated strong lens which clearly has a distinct primary
lens galaxy, but additional galaxies can be seen in and around the Einstein ring. These galaxies are faint and small
in number, but their lensing effects on the source are significant enough that we must ultimately include them in the
lens model.
"""
dataset_name = "lens_x3__source_x1"
dataset_path = path.join("dataset", "group", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    pixel_scales=0.1,
)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Point Source__

The Source's ring is much larger than other examples (> 5.0") and there are clearly additional galaxies in and around
the main lens galaxy. 

Modeling group scale lenses is challenging, because each individual galaxy must be included in the overall lens model. 
For this simple overview, we will therefore model the system as a point source, which reduces the complexity of the 
model and reduces the computational run-time of the model-fit.

Lets load the lens's point-source data, where the brightest pixels of the source are used as the locations of its
centre:
"""
point_dict = al.PointDict.from_json(
    file_path=path.join(dataset_path, "point_dict.json")
)

visuals_2d = aplt.Visuals2D(positions=point_dict.positions_list)

array_plotter = aplt.Array2DPlotter(array=imaging.image, visuals_2d=visuals_2d)
array_plotter.figure_2d()

"""
__PositionsSolver__

Define the position solver used for the point source fitting.
"""
grid = al.Grid2D.uniform(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales
)

positions_solver = al.PositionsSolver(grid=grid, pixel_scale_precision=0.025)

"""
__Model__

We now compose the lens model. For groups there could be many lens and source galaxies in the model. Whereas previous 
examples explicitly wrote the model out via Python code, for group modeling we opt to write it in .json files which
are loaded in this script.

The code below loads a model from a `.json` file created by the script `group/models/lens_x3__source_x1.py`. This 
model includes all three lens galaxies where the priors on the centres have been paired to the brightest pixels in the 
observed image, alongside a source galaxy which is modeled as a point source.
"""
model_path = path.join("scripts", "group", "models")
model_file = path.join(model_path, "lens_x3__source_x1.json")

lenses_file = path.join(model_path, "lenses.json")
lenses = af.Collection.from_json(file=lenses_file)

sources_file = path.join(model_path, "sources.json")
sources = af.Collection.from_json(file=sources_file)

galaxies = lenses + sources

model = af.Collection(galaxies=galaxies)

"""
__Search + Analysis + Model-Fit__

We are now able to model this dataset as a point source, using the exact same tools we used in the point source 
overview.
"""
search = af.DynestyStatic(name="overview_groups")

analysis = al.AnalysisPoint(point_dict=point_dict, solver=positions_solver)

result = search.fit(model=model, analysis=analysis)

"""
__Result__

The result contains information on every galaxy in our lens model:
"""
print(result.max_log_likelihood_instance.galaxies.lens_0.mass)
print(result.max_log_likelihood_instance.galaxies.lens_1.mass)
print(result.max_log_likelihood_instance.galaxies.lens_2.mass)

"""
__Extended Source Fitting__

For group-scale lenses like this one, with a modest number of lens and source galaxies, **PyAutoLens** has all the
tools you need to perform extended surface-brightness fitting to the source's extended emission, including the use
of a pixelized source reconstruction.

This will extract a lot more information from the data than the point-source model and the source reconstruction means
that you can study the properties of the highly magnified source galaxy.

This type of modeling uses a lot of **PyAutoLens**'s advanced model-fitting features which are described in chapters 3
and 4 of the **HowToLens** tutorials. An example performing this analysis to the lens above can be found in the 
notebook `groups/chaining/point_source_to_imaging.ipynb`.

__Wrap Up__

The `group` package of the `autolens_workspace` contains numerous example scripts for performing group-sale modeling 
and simulating group-scale strong lens datasets.
"""
