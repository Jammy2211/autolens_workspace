"""
Overview: Clusters
------------------

The strong lenses we've discussed so far have just a single lens galaxy responsible for the lensing, with a single
source galaxy observed to be lensed. A strong lensing cluster is a system where there are multiple lens galaxies,
deflecting the one or more background sources.

Galaxy clusters range in scale, between the following two extremes:

 - Groups: Strong lenses with a distinct 'primary' lens galaxy and a handful of lower mass galaxies nearby. These
 typically have just one or two lensed sources whose arcs are visible. The Einstein Radii of these systems typically
 range from range 5.0" -> 10.0" (with galaxy scale lenses typically below 5.0").

 - Clusters: These are objects with tens or hundreds of lens galaxies and lensed sources, where the lensed sources
 are all at different redshfits. .

**PyAutoLens** has tools for modeling cluster datasets anywhere between these two extremes, and the example datasets
available throughout the `autolens_workspace` illustrate the different tools that be used for modeling clusters of
different scales.
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
__Group Scale__

In this overview lets begin on the group scale with a simulated strong lens which clearly has a distinct primary
lens galaxy, but additional galaxies can be seen in and around the Einstein ring. These galaxies are faint and small
in number, but their lensing effects on the source are significant enough that we must ultimately include them in the
lens model.
"""
dataset_name = "lens_x3__source_x1"
dataset_path = path.join("dataset", "clusters", dataset_name)

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

array_plotter = aplt.Array2DPlotter(array=imaging.image_a, visuals_2d=visuals_2d)
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

We now compose the lens model. For clusters there could be many hundreds of galaxies in the model. Whereas previous 
examples explicitly wrote the model out via Python code, for cluster modeling we opt to write it in .json files which
are loaded in this script.

The code below loads a model from the .json file `clusters/modeling/models/lens_x3__source_x1.py`. This model includes
all three lens galaxies where the priors on the centres have been paired to thei brightest pixels in the observed image,
alongside a source galaxy which is modeled as a point source.
"""
model_path = path.join("scripts", "clusters", "modeling", "models")
model_file = path.join(model_path, "lens_x3__source_x1.json")

model = af.Collection.from_json(file=model_file)

"""
__Search + Analysis + Model-Fit (Search 1)__

We are now able to model this dataset as a point source, using the exact same tools we used in the point source 
overview.
"""
search_1 = af.DynestyStatic(name="overview_clusters_group")

analysis = al.AnalysisPoint(
    point_dict=point_dict, solver=positions_solver
)

result_1 = search_1.fit(model=model, analysis=analysis)

"""
__Full Image Fitting__

For group-scale lenses like this one, with a modest number of lens and source galaxies, **PyAutoLens** has all the
tools you need to perform extended surface-brightness fitting to the source's extended emission, including the use
of a pixelized source reconstruction.

This will extract a lot more information from the data than the point-source model and the source reconstruction means
that you can study the properties of the highly magnified source galaxy.

This type of modeling uses a lot of **PyAutoLens**'s advanced model-fitting features which are described in chapters 3
and 4 of the **HowToLens** tutorials. An example performing this analysis to the lens above can be found in the 
notebook `clusters/modeling/chaining/lens_x3__source_x1.ipynb`.
"""
