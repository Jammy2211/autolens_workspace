"""
Preprocess: Point Data
======================

To model galaxy clusters with many sources, we typically model them as point sources where the (y,x) locations of the
brightest pixels of each lensed source are used as constraints. Flux and time-delay information may also be included.

This script shows how to create the `PointSourceDict` object for a cluster that is used in a model-fit, alongside the
`Point` models of every lensed source.
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

We will use this to verify that our source positions are aligned with the data.
"""
dataset_path = path.join("..", "sdssj1152p3312")

# image = al.Array2D.from_fits(
#     file_path=path.join(dataset_path, "f160w_image.fits"),
#     hdu=0,
#     pixel_scales=0.03
# )
#
# mat_plot_2d = aplt.MatPlot2D(
#     cmap=aplt.Cmap(vmin=0.0, vmax=0.1)
# )
#
# array_plotter = aplt.Array2DPlotter(
#     array=image.native, mat_plot_2d=mat_plot_2d
# )
# array_plotter.figure_2d()

"""
__Point Source Dict__

We now create the `PointSourceDict` containing the locations of every image of the lensed sources.

We do this by loading it from the catalogue file `source.cat`, like we did for the lenses. 

Each row corresponds to a multiple image of a source galaxy in the cluster and the columns from left to right are as 
follows:

 1) The ID of the source (e.g. 1).
 2) The RA coordinate of the multiple image (e.g. 177.9988491).
 3) The DEC coordinate of the multiple image (e.g. 33.22729236).
 4) The error on the position's x measurement, which is just the image pixel scale (e.g. 0.1).
 5) The error on the position's y measurement, which is just the image pixel scale (e.g. 0.1).
 6) A value of 0.0, for some reason.
 7) The redshift of the source galaxy.
 8) Anotehr 0., for some reason.

Below, we extract each column of this table into Python lists.
"""
def catalogue_to_lists(file):

    with open(file) as f:
        l = f.read().split("\n")

    combined = list(zip(*[item.split(" ") for item in filter(lambda item: item, l)]))
    print(combined)
    return [list(map(int, combined[0]))] + [list(map(float, column)) for column in combined[1:]]

catalogue_file = path.join("dataset", "clusters", "sdssj1152p3312", "source.cat")
catalogue = catalogue_to_lists(file=catalogue_file)

"""
The Brightest Cluster Galaxy (BGC) is used to set the origin of our coordinate system, which the lensed source multiple
imge positions are computed from.

I estimated this centre using the GUI script found in the `preprocess/gui` folder, you may want to use this on your 
cluster dataset!
"""
bcg_centre = (34.305, 24.075)

"""
We next convert the coordinates of each multiple image from RA / DEC to arc-second coordinates in the frame of the 
image, where the origin of the coordinate system is the centre of the BCG.
"""
ra_list = catalogue[1]
dec_list = catalogue[2]

# galaxy_centres = al.Grid2DIrregular.from_ra_dec(ra=ra_list, dec=dec_list, origin=bcg_centre)

multiple_images = [(1.0, 2.0)] * len(ra_list)

"""
The galaxy ids are used for creating the source models.
"""
id_list = catalogue[0]

"""
The galaxy redshift will be used to create the cluster lens model, ensuring that multi-plane ray tracing is properly
accounted for by the lens model.
"""
redshift_list = catalogue[6]

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
We now compose the galaxies as a `Model`, where each source is a `PointSourceChi` meaning that it is fitted as a point
source and its goodness-of-fit is evaluated in the source plane.
"""
galaxies = []

for index, id in enumerate(id_list):

    point = af.Model(al.ps.PointSourceChi)

al.ps.PointSourceChi