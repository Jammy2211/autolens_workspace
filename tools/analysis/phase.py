import autofit as af
import os

# In this example, we'll generate a phase which fits a simple lens + source plane system. Whilst I would generally
# recommend that you write pipelines when using PyAutoLens, it can be convenient to sometimes perform non-linear
# searches in one phase to get results quickly.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

# There is a x2 '/../../' because we are in the 'autolens_workspace/scripts/examples' folder, so we need to move up two
# folders to get to the "autolens_workspace" folder.

# Use this path to explicitly set the config path and output papth
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

# It is convenient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the function to load the imaging dataset below.
dataset_label = "imaging"
dataset_name = "lens_sersic_sie__source_sersic"
pixel_scales = 0.1

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

import autolens as al
import autolens.plot as aplt

# Load the imaging dataset.
imaging = al.imaging.from_fits(
    image_path=dataset_path + "/image.fits",
    psf_path=dataset_path + "/psf.fits",
    noise_map_path=dataset_path + "/noise_map.fits",
    pixel_scales=pixel_scales,
)

# The phase can be passed a mask, which we setup below as a 3.0" circle.
mask = al.mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1, radius=3.0
)

# We can also specify a set of positions, which must be traced within a threshold value or else the mass model is.
positions = al.coordinates.from_file(file_path=dataset_path + "/positions.dat")

# Lets plot the imaging, mask and positions before we run the analysis.
aplt.imaging.subplot_imaging(imaging=imaging, mask=mask, positions=positions)

# We're going to model our lens galaxy using a light profile (an elliptical Sersic) and mass profile
# (a singular isothermal ellipsoid). We load these profiles from the 'light_profiles (lp)' and 'mass_profiles (mp)'.

# To setup our model galaxies, we use a GalaxyModel, which represents a galaxy of which the parameters of its
# associated profiles are variable and fitted for by the analysis.
lens_galaxy_model = al.GalaxyModel(
    redshift=0.5, light=al.lp.EllipticalSersic, mass=al.mp.EllipticalIsothermal
)

source_galaxy_model = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

# To perform the analysis, we set up a phase using a PhaseImaging object, which takes our galaxy models and fits their
# parameters using a non-linear search (in this case, MultiNest).
phase = al.PhaseImaging(
    phase_name="phase_example",
    phase_folders=[dataset_label, dataset_name],
    galaxies=dict(lens=lens_galaxy_model, source=source_galaxy_model),
    optimizer_class=af.MultiNest,
)

# The phase folders and phase name mean the output of these run will be in the directory
# 'autolens_workspace/output/imaging/lens_sie__source_sersic/phase_example'

# You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
# non-linear parameter space faster (if you haven't already, checkout the tutorial '' in howtolens/chapter_2).

phase.optimizer.const_efficiency_mode = True
phase.optimizer.n_live_points = 50
phase.optimizer.sampling_efficiency = 0.5

# We run the phase on the image, print the results and plotters the fit.
result = phase.run(dataset=imaging, mask=mask)

aplt.fit_imaging.subplot_fit_imaging(fit=result.most_likely_fit)
