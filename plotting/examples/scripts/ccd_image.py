import autofit as af
import autolens as al

import os

# In this example, we will load the image of a strong lens from a .fits file and plotters it using the
# function autolens.dataset_label.plotters.array_plotters.plot_array. We will customize the appearance of this figure to
# highlight the features of the image. For more generical plotting tools (e.g. changing the figure size, axis unit_label,
# outputting the image to the hard-disk, etc.) checkout the example in 'autolens_workspace/plotting/examples/structures/masked_structures.py'.

# We will use the image of slacs1430+4105.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# We have included the .fits dataset_label required for this example in the directory
# 'autolens_workspace/output/dataset/imaging/slacs1430+4105/'.

# First, lets setup the path to the .fits file of the image.
dataset_label = "imaging"
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

# We can now use an arrays plotter to plotters the image. Lets first plotters it using the default PyAutoLens settings.
al.plot.array(array=image, title="SLACS1430+4105 Image")

# For a lens like SLACS1430+4105, the lens galaxy's light outshines the background source, making it appear faint.
# we can use a symmetric logarithmic colorbar normalization to better reveal the source galaxy (due to negative values
# in the image, we cannot use a logirithmic colorbar normalization).
al.plot.array(
    array=image,
    title="SLACS1430+4105 Image",
    norm="symmetric_log",
    linthresh=0.05,
    linscale=0.02,
)

# Alternatively, we can use the default linear colorbar normalization and customize the limits over which the colormap
# spans its dynamic range.
al.plot.array(
    array=image, title="SLACS1430+4105 Image", norm="linear", norm_min=0.0, norm_max=0.3
)

psf_path = workspace_path + "/dataset/imaging/" + dataset_name + "/psf.fits"
noise_map_path = workspace_path + "/dataset/imaging/" + dataset_name + "/noise_map.fits"

imaging = al.imaging.from_fits(
    image_path=image_path,
    psf_path=psf_path,
    noise_map_path=noise_map_path,
    pixel_scales=0.03,
)

# These plotters can be customized using the exact same functions as above.
al.plot.imaging.noise_map(imaging=imaging, title="SLACS1430+4105 Noise-Map", norm="log")

# Of course, as we've seen in many other examples, a sub-plotters of the imaging dataset_label can be plotted. This can also take the
# customization inputs above, but it should be noted that the options are applied to all images, and thus will most
# likely degrade a number of the sub-plotters images.
al.plot.imaging.subplot(
    imaging=imaging, norm="symmetric_log", linthresh=0.05, linscale=0.02
)
