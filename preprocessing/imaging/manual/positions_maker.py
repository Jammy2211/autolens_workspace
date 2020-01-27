import autofit as af
import autolens as al
import autolens.plot as aplt
import os

# This tool allows one to input a set of positions of a multiply imaged strongly lensed source, corresponding to a set
# positions / pixels which are anticipated to trace to the same location in the source-plane.

# A non-linear sampler uses these positions to discard the mass-models where they do not trace within a threshold of
# one another, speeding up the analysis and removing unwanted solutions with too much / too little mass.

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
positions = al.coordinates([[(0.8, 1.45), (1.78, -0.4), (-0.95, 1.38), (-0.83, -1.04)]])

# These are the positions for the example lens 'lens_sersic_sie__source_sersic__2'
# positions = al.coordinates([[(0.44, 0.60), (-1.4, -0.41), (0.15, -1.45)]])

# These are the positions for the example lens 'lens_sie__source_sersic__2'
# positions = al.coordinates([[(1.28, -1.35), (-0.5, 0.7)]])

# These are the positions for the example lens 'lens_sie__source_sersic_x2'
# positions = al.coordinates([[(2.16, -1.3), (-0.65, 0.45)]])

# We can infact input multiple lists of positions (commented out below), which corresponds to pixels which are \
# anticipated to map to different multiply imaged regions of the source-plane (e.g. you would need something like \
# spectra to be able to do this)
# Images of source 1           # Images of source 2
# positions = [[(1.0, 1.0), (2.0, 0.5)], [(-1.0, -0.1), (2.0, 2.0), (3.0, 3.0)]]

# Now lets plotters the image and positions, so we can check that the positions overlap different regions of the source.
aplt.array(array=image, positions=positions)

# Now we're happy with the positions, lets output them to the dataset folder of the lens, so that we can load them from a
# .dat file in our pipelines!
positions.output_to_file(file_path=dataset_path + "positions.dat")


# These commented out lines would create the positions for the example_multi_plane dataset.
# lens_name = 'example_multi_plane'
# pixel_scales = 0.05
# positions = [[(0.8, 1.12), (-0.64, 1.13), (1.38, -0.2), (-0.72, -0.83)]]
