import autofit as af
import autolens as al
import autolens.plot as aplt
import autolens.plot as aplt

import os

# In this example, we will demonstrate how the appearance of figures in PyAutoLens can be customized. To do this, we
# will use the the image of the strong lens slacs1430+4105 from a .fits file and plotters it using the
# function autolens.dataset_label.plotters.plotters.plot_array.

# The customization functions demonstrated in this example are generic to any 2D arrays of dataset_label, and can therefore be
# applied to the plotting of noise-maps, PSF's, residual maps, chi-squared maps, etc.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# We have included the .fits dataset_label required for this example in the directory
# 'autolens_workspace/output/dataset/imaging/slacs1430+4105/'.

# First, lets setup the path to the .fits file of the image.
dataset_label = "slacs"
dataset_name = "slacs1430+4105"

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/slacs1430+4105/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)
image_path = dataset_path + "image.fits"

# Now, lets load this arrays as a hyper arrays. A hyper arrays is an ordinary NumPy arrays, but it also includes a pixel
# scale which allows us to convert the axes of the arrays to arc-second coordinates.
image = al.array.from_fits(file_path=image_path, hdu=0, pixel_scales=0.03)

# We can now use an arrays plotter to plotters the arrays. We customize the plotters as follows:

# 1) We make the arrays's figure size bigger than the default size (7,7).

# 2) Because the figure is bigger, we increase the size of the title, x and y labels / ticks from their default size of
#    16 to 24.

# 3) For the same reason, we increase the size of the colorbar ticks from the default value 10 to 20.

plotter = aplt.Plotter(
    figure=aplt.Figure(figsize=(12, 12)),
    labels=aplt.Labels(title="SLACS1430+4105 Image", titlesize=24, ysize=24, xsize=24),
    ticks=aplt.Ticks(ysize=24, xsize=24),
    cmap=aplt.ColorMap(norm="cold", norm_max=0.8),
    cb=aplt.ColorBar(ticksize=20),
)

aplt.array(array=image, plotter=plotter)

# The colormap of the arrays can be changed to any of the standard matplotlib colormaps.

plotter = aplt.Plotter(
    labels=aplt.Labels(title="SLACS1430+4105 Image"), cmap=aplt.ColorMap(cmap="spring")
)

aplt.array(array=image, plotter=plotter)

# We can change the x / y axis unit_label from arc-seconds to kiloparsec, by inputting a kiloparsec to arcsecond conversion
# factor (for SLACS1430+4105, the lens galaxy is at redshift 0.285, corresponding to the conversion factor below).

plotter = aplt.Plotter(
    labels=aplt.Labels(title="SLACS1430+4105 Image"),
    units=aplt.Units(in_kpc=True, conversion_factor=6.2),
)

aplt.array(array=image, plotter=plotter)

# The matplotlib figure can be output to the hard-disk as a png, as follows.

plotter = aplt.Plotter(
    labels=aplt.Labels(title="SLACS1430+4105 Image"),
    output=aplt.Output(
        path=workspace_path + "/plotting/plots/", filename="arrays", format="png"
    ),
)

aplt.array(array=image, plotter=plotter)

# The arrays itself can be output to the hard-disk as a .fits file.

plotter = aplt.Plotter(
    labels=aplt.Labels(title="SLACS1430+4105 Image"),
    output=aplt.Output(
        path=workspace_path + "/plotting/plots/", filename="arrays", format="fits"
    ),
)

aplt.array(array=image, plotter=plotter)
