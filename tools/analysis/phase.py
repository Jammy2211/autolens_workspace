import autofit as af
from autolens.pipeline.phase import phase_imaging
from autolens.data.array import mask as msk
from autolens.model.galaxy import galaxy_model as gm
from autolens.data import ccd
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp
from autolens.data.plotters import ccd_plotters
from autolens.lens.plotters import lens_fit_plotters

import os

# In this example, we'll generate a phase which fits a simple lens + source plane system. Whilst I would generally
# recommend that you write pipelines when using PyAutoLens, it can be convenient to sometimes perform non-linear
# searches in one phase to get results quickly.

# Setup the path to the workspace, using a relative directory name.
workspace_path = '{}/../../'.format(os.path.dirname(os.path.realpath(__file__)))

# There is a x2 '/../../' because we are in the 'workspace/scripts/examples' folder. If you write your own script \
# in the 'workspace/script' folder you should remove one '../', as shown below.
# path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output papth
af.conf.instance = af.conf.Config(
    config_path=workspace_path + 'config', output_path=workspace_path + 'output')

# It is convinient to specify the lens name as a string, so that if the pipeline is applied to multiple images we \
# don't have to change all of the path entries in the load_ccd_data_from_fits function below.
data_type = 'example'
data_name = 'lens_light_and_x1_source'
pixel_scale = 0.1

# Create the path where the data will be loaded from, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

ccd_data = ccd.load_ccd_data_from_fits(
    image_path=data_path + '/image.fits',
    psf_path=data_path + '/psf.fits',
    noise_map_path=data_path + '/noise_map.fits',
    pixel_scale=pixel_scale)

# The phase can be passed a mask, which we setup below as a 3.0" circle.
mask = msk.Mask.circular(
    shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale, radius_arcsec=3.0)

# We can also specify a set of positions, which must be traced within a threshold value or else the mass model is
positions = ccd.load_positions(
    positions_path=data_path + '/positions.dat')

# resampled (see howtolens/chapter_2_lens_modeling/tutorial_7_masking_and_positions.ipynb)
ccd_plotters.plot_ccd_subplot(
    ccd_data=ccd_data, mask=mask, positions=positions)

# We're going to model our lens galaxy using a light profile (an elliptical Sersic) and mass profile
# (a singular isothermal ellipsoid). We load these profiles from the 'light_profiles (lp)' and 'mass_profiles (mp)'.

# To setup our model galaxies, we use the 'galaxy_model' module and GalaxyModel class.
# A GalaxyModel represents a galaxy where the parameters of its associated profiles are
# variable and fitted for by the analysis.
lens_galaxy_model = gm.GalaxyModel(
    redshift=0.5,
    light=lp.EllipticalSersic,
    mass=mp.EllipticalIsothermal)

source_galaxy_model = gm.GalaxyModel(
    redshift=1.0,
    light=lp.EllipticalSersic)

# To perform the analysis, we set up a phase using the 'phase' module (imported as 'ph').
# A phase takes our galaxy models and fits their parameters using a non-linear search (in this case, MultiNest).

# The phase folders and phase name mean the output of these run will be in the directory
# 'workspace/output/example/lens_light_and_x1_source/phase_example'
phase = phase_imaging.LensSourcePlanePhase(
    phase_name='phase_example', phase_folders=[data_type, data_name],
    lens_galaxies=dict(
        lens=lens_galaxy_model),
    source_galaxies=dict(
        source=source_galaxy_model),
    optimizer_class=af.MultiNest)

# You'll see these lines throughout all of the example pipelines. They are used to make MultiNest sample the \
# non-linear parameter space faster (if you haven't already, checkout the tutorial '' in howtolens/chapter_2).

phase.optimizer.const_efficiency_mode = True
phase.optimizer.n_live_points = 50
phase.optimizer.sampling_efficiency = 0.5

# We run the phase on the image, print the results and plot the fit.
result = phase.run(data=ccd_data)

lens_fit_plotters.plot_fit_subplot(fit=result.most_likely_fit)