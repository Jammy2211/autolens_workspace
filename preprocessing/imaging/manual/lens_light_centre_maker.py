import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# This tool allows one to input the lens light centre(s) of a strong lens(es), which can be used as a fixed value in
# pipelines.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the positions are stored in e.g,
# the positions will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/positions.dat'.
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"

# Create the path where the mask will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
pixel_scales = 0.1

# First, load the imaging dataset, so that the positions can be plotted over the strong lens image.
image = al.array.from_fits(
    file_path=dataset_path + "image.fits", pixel_scales=pixel_scales
)

# Now, create a set of positions, which is a python list of (y,x) values.
lens_light_centre = al.coordinates([[(0.0, 0.0)]])

# Now lets plotters the image and positions, so we can check that the positions overlap different regions of the source.
aplt.array(array=image, light_profile_centres=lens_light_centre)

# Now we're happy with the positions, lets output them to the dataset folder of the lens, so that we can load them from a
# .dat file in our pipelines!
lens_light_centre.output_to_file(file_path=dataset_path + "lens_light_centre.dat")
