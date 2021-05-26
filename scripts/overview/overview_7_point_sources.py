"""
Overview: Point Sources
-----------------------

So far, the PyAutoLens overview has shown strongly lensed galaxies, whose extended surface brightness is lensed into
the awe-inspiring giant arcs and Einstein rings we see in high quality telescope imaging. There are many lenses where
the background source is not extended but is instead a point-source, for example strongly lensed quasars and supernovae.

For these objects, we do not want to model the source using a `LightProfile` which implicitly assumes an extended
surface brightness distribution. Instead, we assume that our source is a point source with a centre (y,x). Our
ray-tracing calculations no longer trace extended light rays from the source plane to the image-plane, but instead
now find the locations the point-source's multiple images appear in the image-plane.

Finding the multiple images of a mass model given a (y,x) coordinate in the source plane is an iterative problem
performed in a very different way to ray-tracing a `LightProfile`. In this example, we introduce **PyAutoLens**`s
`PositionSolver`, which does exactly this and thus makes the analysis of strong lensed quasars, supernovae and
point-like source's possible in **PyAutoLens**! we'll also show how these tools allow us to compute the flux-ratios
and time-delays of the point-source.
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
__Lensed Point Source__

To begin, we will create an image of strong lens using a simple `EllIsothermal` mass model and source with an
`EllExponential` light profile. Although we are going to show how **PyAutoLens**`s positional analysis tools 
model point-sources, showing the tools using an extended source will make it visibly clearer where the multiple 
images of the point source are!

Below, we set up a `Tracer` using a `Grid2D`, `LightProfile`, `MassProfile` and two `Galaxy`'s. These objects are 
introduced in the `lensing.py` example script, so if it is unclear what they are doing you should read through that
example first before continuing!
"""
grid = al.Grid2D.uniform(
    shape_native=(100, 100),
    pixel_scales=0.05,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

isothermal_mass_profile = al.mp.EllIsothermal(
    centre=(0.001, 0.001), einstein_radius=1.0, elliptical_comps=(0.0, 0.111111)
)

exponential_light_profile = al.lp.EllExponential(
    centre=(0.07, 0.07),
    elliptical_comps=(0.2, 0.0),
    intensity=0.05,
    effective_radius=0.2,
)

lens_galaxy = al.Galaxy(redshift=0.5, mass=isothermal_mass_profile)

source_galaxy = al.Galaxy(redshift=1.0, light=exponential_light_profile)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
Lets plot the image of our strongly lensed source galaxy. By eye, we can clearly see there are four multiple images 
located in a cross configuration, which are the four (y,x) multiple image coordinates we want our positional solver
to find! 
"""
tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
tracer_plotter.figures_2d(image=True)

"""
__Point Source__

The image above visually illustrates where the source's light traces in the image-plane. Lets now treat this source
as a point source, by setting up a source galaxy and `Tracer` using the `Point` class. 
"""
point_source = al.ps.Point(centre=(0.07, 0.07))

source_galaxy = al.Galaxy(redshift=1.0, point=point_source)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""
__Solving the Lens Equation__

For a `Point`, our goal is to find the (y,x) coordinates in the image-plane that directly map to the centre
of the `Point` in the source plane. In this example, we therefore need to compute the 4 image-plane that map
directly to the location (0.07", 0.07") in the source plane.

This is often referred to as 'solving the lens equation' in the literature.

This is an iterative problem that requires us to use the `PositionsFinder`. 
"""
solver = al.PositionsSolver(
    grid=grid,
    pixel_scale_precision=0.001,
    upscale_factor=2,
    distance_from_source_centre=0.01,
)

"""
We now pass the `Tracer` to the solver. This will then find the image-plane coordinates that map directly to the
source-plane coordinate (0.07", 0.07").
"""
positions = solver.solve(lensing_obj=tracer, source_plane_coordinate=(0.07, 0.07))

grid_plotter = aplt.Grid2DPlotter(grid=positions)
grid_plotter.figure_2d()

"""
You might be wondering why don't we use the image of the lensed source to compute our multiple images. Can`t we just 
find the pixels in the image whose flux is brighter than its neighboring pixels? 

Although this might work, for positional modeling we want to know the (y,x) coordinates of the multiple images at a 
significantly higher precision than the grid we see the image on. In this example, the grid has a pixel scale of 0.05",
however we determine our multiple image positions at scales of 0.01"!

__Lens Modeling__

**PyAutoLens** has full support for modeling strong lens datasets as a point-source. This might be used for analysing
strongly lensed quasars or supernovae, which are so compact we do not observe their extended emission.

To perform point-source modeling, we first create a ``PointDataset`` containing the image-plane (y,x) positions
of each multiple image and their noise values (which would be the resolution of the imaging data they are observed). 

The positions below correspond to those of an `EllIsothermal` mass model.
"""
point_dataset = al.PointDataset(
    name="point_0",
    positions=al.Grid2DIrregular(
        [[1.1488, -1.1488], [1.109, 1.109], [-1.109, -1.109], [-1.1488, 1.1488]]
    ),
    positions_noise_map=al.ValuesIrregular([0.05, 0.05, 0.05, 0.05]),
)

"""
__Point Source Dictionary__

In this simple example we model a single point source, which might correspond to one lensed quasar or supernovae.
However, **PyAutoLens** supports model-fits to datasets with many lensed point-sources, for example in galaxy clusters.

Each point source dataset is therefore passed into a `PointDict` object before the model-fit is performed. For 
this simple example only one dataset is passed in, but in the galaxy-cluster examples you'll see this object makes it
straightforward to model datasets with many lensed sources.
"""
point_dict = al.PointDict(point_dataset_list=[point_dataset])

"""
We can print the `positions` of this dictionary and dataset, as well as their noise-map values.
"""
print("Point Source Dataset Name:")
print(point_dict["point_0"].name)
print("Point Source Multiple Image (y,x) Arc-second Coordinates:")
print(point_dict["point_0"].positions.in_list)
print("Point Source Multiple Image Noise-map Values:")
print(point_dict["point_0"].positions_noise_map.in_list)

"""
__Naming__

Every point-source dataset in the `PointDict` has a name, which in this example was `point_0`. This `name` pairs 
the dataset to the `Point` in the model below. Because the name of the dataset is `point_0`, the 
only `Point` object that is used to fit it must have the name `point_0`.

If there is no point-source in the model that has the same name as a `PointDataset`, that data is not used in
the model-fit. If a point-source is included in the model whose name has no corresponding entry in 
the `PointDataset` **PyAutoLens** will raise an error.

In this example, where there is just one source, name pairing appears unecessary. However, point-source datasets may
have many source galaxies in them, and name pairing is necessary to ensure every point source in the lens model is 
fitted to its particular lensed images in the `PointDict`!

__Model__

We first compose the model, in the same way described in the `modeling.py` overview script:
"""
lens_galaxy_model = af.Model(
    al.Galaxy, redshift=0.5, bulge=al.lp.EllSersic, mass=al.mp.EllIsothermal
)

source_galaxy_model = af.Model(al.Galaxy, redshift=1.0, point_0=al.ps.Point)

model = af.Collection(lens=lens_galaxy_model, source=source_galaxy_model)

"""
__Non-linear Search__

We again choose the non-linear search `dynesty` (https://github.com/joshspeagle/dynesty).
"""
search = af.DynestyStatic(name="overview_point_source")

"""
__Analysis__

Whereas we previously used an `AnalysisImaging` object, we instead use an `AnalysisPoint` object which fits the
lens model in the correct way for a point source dataset.

This includes mapping the `name`'s of each dataset in the `PointDict` to the names of the point sources in the
lens model.
"""
analysis = al.AnalysisPoint(point_dict=point_dict, solver=solver)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

The results can be found in the `output/overview_point_source` folder in the `autolens_workspace`.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The **PyAutoLens** visualization library and `FitPoint` object includes specific methods for plotting the results.
"""

"""
__Wrap Up__

The `point_source` package of the `autolens_workspace` contains numerous example scripts for performing point source
modeling to datasets where there are only a couple of lenses and lensed sources, which fall under the category of
'galaxy scale' objects.

This also includes examples of how to add and fit other information that are observed by a point-source source,
for example the flux of each image.

If you wish to model systems with many lens galaxy and sources, e.g. galaxy clusters, checkout the `galaxy_clusters.py`
overview script.
"""
