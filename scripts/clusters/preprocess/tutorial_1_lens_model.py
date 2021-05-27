"""
Preprocess: Galaxy Catalogue
============================

To model galaxy clusters comprised of many lens galaxies, the lens model is composed of three distinct objects:

 1) Galaxies that are sufficiently massive or near a lensed arc such that their mass is modeled explicitly (e.g.
 the Brighest Cluster Galaxy).

 2) The dark matter component(s) of the galaxy cluster.

 3) The many tens or hundreds of smaller galaxies that contribute collectively to the lensing, that are modeled
 assuming that their mass traces their light via a scaling relation.

This tutorial shows how to compose a lens model including all three components.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import json
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load and plot the image of our example strong lens cluster, sdssj1152p3312. 

We will use this to verify that our strong lens galaxy centres are aligned with the data.
"""
dataset_path = path.join("..", "sdssj1152p3312")

image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "f160w_image.fits"), hdu=0, pixel_scales=0.03
)

mat_plot_2d = aplt.MatPlot2D(cmap=aplt.Cmap(vmin=0.0, vmax=0.1))

array_plotter = aplt.Array2DPlotter(array=image.native, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

"""
__Redshift__

The redshift of the galaxy cluster, and therefore all its members, it used by the lens model and therefore must
be set. 
"""
redshift_lens = 0.5

"""
__Brightest Cluster Galaxy (BCG) + Dark Matter__

We are now going to create the model of the brightest cluster galaxy (BCG), which we wish to fit for explicitly in 
the lens model.

We will model the BCG using an `EllIsothermal` model. The centre of the BCG will be the origin of our coordinate system,
so we do not need to update the priors on its centre (which are centred around (0.0", 0.0") by default.
"""
mass = af.Model(al.mp.EllIsothermal)
bcg = af.Model(al.Galaxy, mass=mass)

"""
We next create the model of the cluster's dark matter halo, which we fit explicitly alongside the BCG.

We model the cluster using a `SphNFWMCRLudlow` profile, which sets the mass of the dark matter halo (parameterized 
using the `mass_at_200` times the critical density of the Universe via the mass-concentration relation of Ludlow et al.

This conversion requires the redshifts of the lens and a source, we use a fiducial source at redshift 2.0.
"""
dark = af.Model(al.mp.SphNFWMCRLudlow)
dark.mass_at_200 = af.LogUniformPrior(lower_limit=1.0e14, upper_limit=1.0e16)
dark.redshift_object = redshift_lens
dark.redshift_source = 2.0

halo = af.Model(al.Galaxy, redshift=redshift_lens, dark=dark)

"""
__Scaling Relation__

To model galaxy clusters comprised of many lens galaxies, we need to use a galaxy catalogue (e.g. from Source
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
 
Below, we extract each column of this table into Python lists.
"""


def catalogue_to_lists(file):

    with open(file) as f:
        l = f.read().split("\n")

    combined = list(zip(*[item.split("    ") for item in filter(lambda item: item, l)]))
    return [list(map(int, combined[0]))] + [
        list(map(float, column)) for column in combined[1:]
    ]


catalogue_file = path.join("dataset", "clusters", "sdssj1152p3312", "lens.cat")
catalogue = catalogue_to_lists(file=catalogue_file)

"""
The galaxy catalogue omits the Brightest Cluster Galaxy (BGC) of the cluster, because it is explicitly included in
the lens model for this cluster (as opposed to the galaxies in the galaxy catalogue whose mass is assumed to lie on the
luminosity-to-mass scaling relation).

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

# galaxy_centres = al.Grid2DIrregular.from_ra_dec(ra=ra_list, dec=dec_list, origin=bcg_centre)

galaxy_centres = [(1.0, 2.0)] * len(ra_list)

"""
The galaxy magnitudes will be used to create the cluster lens model.
"""
magnitude_list = catalogue[6]

"""
We now plot the galaxy centres on the image, to make sure the conversion has placed them in the right place (E.g.
over the top of every galaxy!).
"""
# visuals_2d = aplt.Visuals2D(mass_profile_centres=galaxy_centres)
#
# array_plotter = aplt.Array2DPlotter(
#     array=image.native, mat_plot_2d=mat_plot_2d, visuals_2d=visuals_2d
# )
# array_plotter.figure_2d()

"""
We now want compose these galaxies into `Model`'s which are related to a mass-to-light scaling relation.

We want this model to be tied to a scaling relation that relates the mass of each galaxy to the magnitude contained in
the catalogue. We therefore set up the scaling relation as a model, noting that its `gradient` and `intercept` will
be free parameters that are fitted for when we perform lens modeling on the cluster. 
"""
mass_light_relation = af.Model(al.sr.MassLightRelation)
mass_light_relation.gradient = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)
mass_light_relation.intercept = af.UniformPrior(lower_limit=0.0, upper_limit=10.0)

"""
Below, we create a `SphIsothermalMLR` mass model for every galaxy. The `SphIsothermalMLR` is a special form of the
`SphIsothermal` model where its parameters are specifically tied to the `MassLightRelation`. 

When creating each mass model: 

 1) The centre is fixed to its value in the catalogue.
 2) The magnitude is fixed to its value in the catalogue.
 3) The parameters of the scaling relation above are linking to each profile.
 
By linking the scaling relation model to each mass model, this ensures that when we perform model-fitting the mass
of every individual galaxy is set using its magnitude and the scaling relation.
"""
galaxies = [bcg, dark]

for index in range(len(galaxy_centres)):

    mass = af.Model(
        al.sr.SphIsothermalMLR,
        centre=galaxy_centres[index],
        magnitude=magnitude_list[index],
        relation=mass_light_relation,
    )

    galaxy = af.Model(al.Galaxy, redshift=redshift_lens, mass=mass)

    galaxies.append(galaxy)

"""
__Model__

We now package the BCG, dark matter halo, scaling relation and galaxies on the relation up into our overall lens model.

If we print it, we see it consists of over 70 model galaxy objects!
"""
model = af.Collection(scaling_relation=mass_light_relation, galaxies=galaxies)

print(model)

"""
We now write the model to a .json file, so it can be loaded in our model-fiting script.
"""
model_path = path.join("scripts", "clusters", "modeling", "models")
model_file = path.join(model_path, "cluster.json")

with open(model_file, "w+") as f:
    json.dump(model.dict, f, indent=4)

"""
Finish.
"""
