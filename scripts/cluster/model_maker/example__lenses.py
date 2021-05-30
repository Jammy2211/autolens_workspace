"""
Model Makers: Cluster Lenses
============================

For cluster modeling, there can be tens or hundreds of lens and source galaxies. Manually writing the lens model of
each in the Python scripts we used to perform lens model becomes unfeasible and it better for us to manage model
composition in a separate file and store them in .json files.

Furthermore, for clusters, it is unfeasible to manually write out the model of each object, even in this separate
Python script. This example interfaces with a galaxy catalogue file from SourceExtractor
https://sextractor.readthedocs.io/ to compose the model that is fitted in the file `cluster/modeling/example.ipynb`.

__Lens Model__

To model galaxy clusters comprised of many lenses, the lens model is composed of three distinct object types:

 1) lenses that are sufficiently massive or near a lensed arc that their mass is modeled explicitly (e.g.
 the Brightest Cluster Galaxy).

 2) The dark matter component(s) of the galaxy cluster.

 3) Tens or hundreds of smaller lenses that contribute collectively to the lensing, but are too low mass to model
 individually. These are modeled collectively via a scaling relation that assumes light (or some other galaxy property)
 traces mass.

This tutorial shows how to compose a lens model including all three components, and write this model to a `.json` file
so it can be loaded in a model-fitting script.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import numpy as np
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

First, lets load and plot the image of our example strong lens cluster, cluster. 

We will use this to verify that our strong lens galaxy centres are aligned with the data.
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
__Redshift__

The redshift of the galaxy cluster, and therefore all of its member lenses, is used by the lens model and therefore 
must be set. 
"""
redshift_lens = 0.5

"""
__Brightest Cluster Galaxy (BCG)__

We next create the model of the brightest cluster galaxy (BCG), which will be fitted for individually in the lens 
model using an `EllIsothermal` model. This can easily be extended to include multiple lenses in the cluster.

The centre of the BCG will be the origin of our coordinate system, so we do not need to update the priors on its 
centre (which are centred around (0.0", 0.0") by default.
"""
mass = af.Model(al.mp.EllIsothermal)
bcg = af.Model(al.Galaxy, mass=mass)

"""
__Dark Matter__

We next create the model of the cluster's dark matter halo, which we fit individually alongside the BCG.

We model the cluster using a `SphNFWMCRLudlow` profile, which sets the mass of the dark matter halo (parameterized 
using the `mass_at_200` times the critical density of the Universe) via the mass-concentration relation of Ludlow et al.

This conversion requires the redshifts of the lens and a source and we assume a fiducial source at redshift 2.0.
"""
dark = af.Model(al.mp.SphNFWMCRLudlow)
dark.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e13, upper_limit=1.0e16)
dark.redshift_object = redshift_lens
dark.redshift_source = 2.0

halo = af.Model(al.Galaxy, redshift=redshift_lens, dark=dark)

"""
__Scaling Relation__

To model galaxy clusters comprised of many lens lenses, we need to use a galaxy catalogue (e.g. from Source
Extractor https://sextractor.readthedocs.io/) to compose a strong lens model with every galaxy in the cluster and
convert it to a **PyAutoLens** lens model.

Checkout the file `lens.cat`, which is a catalogue from SExtractor. 

Each row corresponds to a galaxy in the cluster and the columns from left to right are as follows:

 1) The ID of the galaxy (e.g. 2).
 2) The RA coordinate of the galaxy (e.g. 177.94166154725815).
 3) The DEC coordinate of the galaxy (e.g. 33.23750122888889).
 4) The semi-major axis of the galaxy ellipticity estimate (e.g. 0.000096).
 5) The semi-minor axis of the galaxy ellipticity estimate (e.g. 0.000078).
 6) The positon angle theta of the ellipse (e.g. -85.400000).
 7) The magnitude of the galaxy (e.g. 18.062900).
 
Below, we use the method `lens_catalogue_to_lists` in `cluster_util` to extract each column of this table into 
Python lists. 

(Your catalogue files may well have a different structure and format to the example one used in this tutorial. You may
need to write your own `lens_catalogue_to_lists` function to do this.)
"""
catalogue_file = path.join("dataset", "clusters", "cluster", "lens.cat")
catalogue = cluster_util.lens_catalogue_to_lists(file=catalogue_file)

"""
__Coordinate System__

The galaxy catalogue omits the Brightest Cluster Galaxy (BGC) of the cluster, given that it is individually included in
the lens model for this cluster (as opposed to the lenses in the galaxy catalogue whose mass is assumed to lie on the
luminosity-to-mass scaling relation). If other lenses were modeled individually they too should be removed from the
galaxy catalogue.

We therefore manually input the arcsecond central coordinates of the BCG, which will act as the origin of our
lens model's coordinate system. I estimated this centre using the GUI script found in the `preprocess/gui` folder, 
you may want to use this on your cluster dataset!
"""
bcg_centre = (34.305, 24.075)

"""
We next convert the coordinates of each galaxy from RA / DEC to arc-second coordinates in the frame of the image,
where the origin of the coordinate system is the centre of the BCG.
"""
ra_list = catalogue[1]
dec_list = catalogue[2]

galaxy_centres = [(1.0, 2.0)] * len(ra_list)
galaxy_centres = al.Grid2DIrregular(grid=galaxy_centres)

"""
The galaxy luminosities in (magnitude units) will be used to create the cluster lens model.
"""
luminosity_list = catalogue[6]

"""
We now plot the galaxy centres on the image, to make sure the conversion has placed them in the right place (E.g.
over the top of every galaxy!).
"""
visuals_2d = aplt.Visuals2D(mass_profile_centres=galaxy_centres)

array_plotter = aplt.Array2DPlotter(
    array=image.native, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
)
array_plotter.figure_2d()

"""
This model is tied to a scaling relation that relates the mass of each galaxy to its luminosity, which was estimated
when constructing the catalogue. We therefore set up the scaling relation as a model, where this relation is of the
form: 

einstein_radius = einstein_radius_factor (luminosity / luminsoity_factor) ** 0.5

`einstein_radius_factor` and `luminosity_factor` are free parameters in the scaling relation
which control the scaling. They are called `gradient` and `denominator` respectively.
"""
mass_light_relation = af.Model(al.sr.MassLightRelation)
mass_light_relation.gradient = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
mass_light_relation.denominator = af.UniformPrior(lower_limit=15.0, upper_limit=20.0)
mass_light_relation.power = 0.5

"""
We can now create a `SphIsothermalMLR` mass model for every galaxy. The `SphIsothermalMLR` is a special form of the
`SphIsothermal` model where its parameters are specifically tied to the `MassLightRelation`. 

When creating each mass model: 

 1) The centre is fixed to its value in the catalogue.
 2) The magnitude is fixed to its value in the catalogue.
 3) The the scaling relation above is passed to the profile, linking it to the scaling relation.
 
By linking the scaling relation model to each mass model, this ensures that when we perform model-fitting the einstein
radius (and therefore mass) of every individual galaxy is set using its luminosity via the scaling relation.
"""
lenses = [bcg, dark]

for index in range(len(galaxy_centres)):

    mass = af.Model(
        al.sr.SphIsothermalMLR,
        relation=mass_light_relation,
        luminosity=luminosity_list[index],
        centre=galaxy_centres.in_list[index],
    )

    galaxy = af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)

    lenses.append(galaxy)

"""
__Model__

We now create the overall model from the BCG, dark matter halo, scaling relation and lenses on the scaling relation.

If we print it, we see it consists of over 70 model galaxy objects!
"""
lenses = af.Collection(scaling_relation=mass_light_relation, lenses=lenses)

print(lenses)

"""
We now write the model to a .json file, so it can be loaded in our model-fitting script.
"""
model_path = path.join("scripts", "clusters", "models", dataset_name)
os.makedirs(model_path, exist_ok=True)

model_file = path.join(model_path, "lenses.json")

with open(model_file, "w+") as f:
    json.dump(lenses.dict, f, indent=4)

"""
Finish.
"""
