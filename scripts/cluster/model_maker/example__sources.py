"""
Model Maker: Cluster Sources
============================

For group & cluster modeling, there can be tens or hundreds of lens and source galaxies. Manually writing the lens
model of each in the Python scripts we used to perform lens model becomes unfeasible and it better for us to manage
model composition in a separate file and store them in .json files.

Furthermore, for clusters, it is unfeasible to manually write out the model of each object, even in this separate
Python script. This example interfaces with a galaxy catalogue file from SourceExtractor
https://sextractor.readthedocs.io/ to compose the model that is fitted in the file `cluster/modeling/example.ipynb`.

__Source Dataset + Model__

Galaxy clusters often contain many lensed background sources, which we typically model as point sources where the (y,x)
locations of the brightest pixels of each lensed source are used as constraints. Flux and time-delay information may
also be included, and **PyAutoLens** includes tools for fitting the imaging data directly, albeit this always builds on
an initial lens model based on point-source fitting.

This script shows how to create the `PointDict` object for a cluster that is used in a model-fit, alongside the
`Point` models of every lensed source.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path
import json
import autofit as af
import autolens as al
import autolens.plot as aplt
import cluster_util

"""
__Downloading Data__

The **PyAutoLens** example cluster datasets are too large in filesize to distribute with the autolens workspace.

They are therefore stored in a separate GitHub repo:

 https://github.com/Jammy2211/autolens_cluster_datasets

Before running this script, make sure you have downloaded the example datasets and moved them to the folder 
`autolens_workspace/dataset/clusters`.

__Dataset__

First, lets load and plot the image of our example strong lens cluster, sdssj1152p3312. 

We will use this to verify that our source positions are aligned with the data.
"""
dataset_name = "cluster"
dataset_path = path.join("dataset", "clusters", dataset_name)

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "f160w_image.fits"), hdu=0, pixel_scales=0.03
)

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=0.1))

array_plotter = aplt.Array2DPlotter(array=image.native, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

"""
__Coordinate System__

The Brightest Cluster Galaxy (BGC) is used to set the origin of our coordinate system, which the lensed source multiple
imge positions are computed from.

I estimated this centre using the GUI script found in the `preprocess/gui` folder, you may want to use this on your 
cluster dataset!
"""
bcg_centre = (34.305, 24.075)

"""
__Point Source Dict__

We first create the `PointDict` containing the locations of every multiple image of every lensed source.

To begin, we load information on the source galaxies and their multiple images from the catalogue file `source.cat`, 
which is similar to the `lens.cat` file we used in the previous tutorial.

Each row corresponds to a multiple image of a source galaxy in the cluster, with spaces between each group of rows 
grouping the different source galaxies. The first column gives the unique id of every source galaxy, which will be used
to group the multiple images in the `PointDict` object on a per source galaxy basis. 
  
The columns from left to right are as  follows:

1) The ID of the source (e.g. 1).
2) The RA coordinate of the multiple image (e.g. 177.9988491).
3) The DEC coordinate of the multiple image (e.g. 33.22729236).
4) The error on the position's x measurement, which is just the image pixel scale (e.g. 0.1).
5) The error on the position's y measurement, which is just the image pixel scale (e.g. 0.1).
6) A value of 0.0, for some reason.
7) The redshift of the source galaxy.
8) Another 0., for some reason.

Below, we use the method `soruce_catalogue_to_lists` in `cluster_util` to extract each column of this table into 
Python lists. 

(Your catalogue files may well have a different structure and format to the example one used in this tutorial. You may
need to write your own `source_catalogue_to_lists` function to do this.)
"""
catalogue_file = path.join("dataset", "clusters", "cluster", "source.cat")
catalogue = cluster_util.source_catalogue_to_lists(file=catalogue_file)

"""
We next convert the coordinates of each multiple image from RA / DEC to arc-second coordinates in the frame of the 
image, where the origin of the coordinate system is the centre of the BCG.
"""
ra_list = catalogue[1]
dec_list = catalogue[2]

# galaxy_centres = al.Grid2DIrregular.from_ra_dec(ra=ra_list, dec=dec_list, origin=bcg_centre)

multiple_image_list = [(1.0, 2.0)] * len(ra_list)
multiple_image_list = al.Grid2DIrregular(grid=multiple_image_list)

"""
The galaxy ids are used for creating the `PointDict` data and the models for each source galaxy.
"""
id_list = catalogue[0]

"""
We now use the unique ids to create:
 
- A list of every source galaxy id.
- A list of list of the multiple images of every source. 

This uses methods in the `cluster_util` package.

By restructuring the catalogue into lists / list of lists that are grouped on a per source basis, this will be it 
simpler to create the `PointDict` and create the cluster's lens model.
"""
id_per_source = cluster_util.list_per_source_from(value_list=id_list)

multiple_image_list_per_source = cluster_util.list_of_lists_per_source_from(
    id_list=id_list, value_list=multiple_image_list
)

"""
We can now construct the `PointDataset` of every source in our galaxy catalogue, which contains the multiple image 
(y,x) coordinates of each galaxy.

Each `PointDataset` has a `name`, for which we use the default **PyAutoFit** convention of `point_id`. When a lens 
model is used to fit each `PointDataset`, this name is paired with the name of `Point` model and we must therefore 
ensure every source galaxy in our `PointDataset` has a corresponding component in the model.

For the noise of every position, we use the `pixel-scale` of the observed image. Every individual position has its
own noise-map value, so we use this pixel-scale to construct a value of every corresponding position for each source.
"""
point_dataset_list = []

for id, multiple_images in zip(id_per_source, multiple_image_list_per_source):

    total_images = len(multiple_images)

    point_dataset = al.PointDataset(
        name=f"point_{id}",
        positions=multiple_images,
        positions_noise_map=al.ValuesIrregular(
            values=total_images * [image.pixel_scales[0]]
        ),
    )

    point_dataset_list.append(point_dataset)

"""
The `PointDict` is a dictionary representation of every `PointDataset`. This is the data object that is passed 
to **PyAutoLens** perform the model-fit.

Below, we create the `PointDict` and write it to a `.json` file so it can be loaded in our modeling script.
"""
point_dict = al.PointDict(point_dataset_list=point_dataset_list)

point_dict.output_to_json(
    file_path=path.join(dataset_path, "point_dict.json"), overwrite=True
)

"""
We now plot the positions of the mulitple images on the image, to make sure the conversion has placed them in the 
right place (e.g. over the top of every galaxy!).
"""
# visuals_2d = aplt.Visuals2D(mass_profile_centres=galaxy_centres)
#
# array_plotter = aplt.Array2DPlotter(
#     array=image.native, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
# )
# array_plotter.figure_2d()

"""
__Source Model__

The redshift of every source galaxy is used to create their entries in the lens model, as it ensures that multi-plane 
ray tracing is properly accounted for.

Below, we create a list of the redshifts per source galaxy.
"""
redshift_list = catalogue[6]

redshift_per_source = cluster_util.list_per_source_from(value_list=redshift_list)


"""
We now compose the lens model. For each source galaxy we create a `Model` of its corresponding point-source, which we
choose here to be a `PointSourceChi`. This means that it is fitted as a point source and its goodness-of-fit is 
evaluated in the source plane. Other models in the `point_source` module can be used to change this behaviour.

We would normally name each component in our model (e.g. `point_0` below) as follows:

`source_galaxy = af.Model(al.Galaxy, redshift=redshift, point_0=point)`

However, we want to use each `id` to create our source galaxies below. We therefore instead use the `setattr` method to 
achieve the same effect.

Each source galaxy in our model is stored in a dictionary, with the name of each galaxy following the syntax `galaxy_id`.
"""
sources = {}

for id, redshift in zip(id_per_source, redshift_per_source):

    point = af.Model(al.ps.PointSourceChi)

    source_galaxy = af.Model(al.Galaxy, redshift=redshift)
    setattr(source_galaxy, f"point_{id}", point)

    sources[f"source_{id}"] = source_galaxy

print(sources)

"""
We now convert this dictionary of `Model` sources to a `Collection1 and write it to a `.json` file, so we can load it
in our model fitting script:
"""
sources = af.Collection(sources)

model_path = path.join("scripts", "clusters", "models", dataset_name)
os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "sources.json")

with open(model_file, "w+") as f:
    json.dump(sources.dict, f, indent=4)

"""
Finish.
"""
