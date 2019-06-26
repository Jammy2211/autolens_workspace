import autofit as af
from autolens.data import ccd
from autolens.model.galaxy import galaxy as g
from autolens.model.profiles import light_profiles as lp
from autolens.model.profiles import mass_profiles as mp

import os

from workspace.tools.data_making.samples import tools

#### UNFINISHE CODE ####

# This tool creates a sample of strong lenses with no lens light component. These lenses use a SIE mass profile and a
# source galaxy generated with x1 Sersic profile. This sample is used to test pipelines and runners.

# The 'data name' is the name of the data folder and 'data_name' the folder the data is stored in, e.g:

# The image will be output as '/workspace/data/data_type/data_name/image.fits'.
# The noise-map will be output as '/workspace/data/data_type/data_name/lens_name/noise_map.fits'.
# The psf will be output as '/workspace/data/data_type/data_name/psf.fits'.

# Setup the path to the workspace, using a relative directory name.
workspace_path = '{}/../../../../'.format(os.path.dirname(os.path.realpath(__file__)))

# (these files are already in the workspace and are remade running this script)
data_imaging_type = 'Euclid'
data_lens_light = 'no_lens_light'
data_lens_type = 'sie_shear_x1_source'

data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', 'samples', data_lens_light, data_lens_type])

# The pixel scale of the image to be simulated
ccd_shape = (120, 120)
pixel_scale = 0.1
sub_grid_size = 1
plot_ccd = True

# Simulate a simple Gaussian PSF for the image.
psf = ccd.PSF.from_gaussian(shape=(11, 11), sigma=0.1, pixel_scale=pixel_scale)

###### SIE + SHEAR LENS 1 #######

data_name = 'sie_shear_x1_source_1'
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.6, axis_ratio=0.7, phi=45.0),
                       shear=mp.ExternalShear(magnitude=0.05, phi=90.0))

source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.1, 0.1), axis_ratio=0.8, phi=60.0,
                                                   intensity=0.3, effective_radius=1.0, sersic_index=2.5))

tools.from_data_lens_and_source_galaxy_simulate_and_output_ccd_data(data_path=data_path, data_name=data_name,
    ccd_shape=ccd_shape, pixel_scale=pixel_scale, psf=psf, lens_galaxy=lens_galaxy, source_galaxy=source_galaxy,
    sub_grid_size=sub_grid_size, plot_ccd=plot_ccd)

###### SIE + SHEAR LENS 2 #######

data_name = 'sie_shear_x1_source_2'
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.2, axis_ratio=0.9, phi=105.0),
                       shear=mp.ExternalShear(magnitude=0.1, phi=145.0))

source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.3, 0.3), axis_ratio=0.6, phi=120.0,
                                                   intensity=0.3, effective_radius=2.0, sersic_index=1.5))

tools.from_data_lens_and_source_galaxy_simulate_and_output_ccd_data(data_path=data_path, data_name=data_name,
    ccd_shape=ccd_shape, pixel_scale=pixel_scale, psf=psf, lens_galaxy=lens_galaxy, source_galaxy=source_galaxy,
    sub_grid_size=sub_grid_size, plot_ccd=plot_ccd)

###### SIE + SHEAR LENS 3 #######

data_name = 'sie_shear_x1_source_3'
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=0.8, axis_ratio=0.9, phi=10.0),
                       shear=mp.ExternalShear(magnitude=0.02, phi=30.0))

source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(-0.2, 0.4), axis_ratio=0.4, phi=160.0,
                                                   intensity=0.1, effective_radius=0.2, sersic_index=2.5))

tools.from_data_lens_and_source_galaxy_simulate_and_output_ccd_data(data_path=data_path, data_name=data_name,
    ccd_shape=ccd_shape, pixel_scale=pixel_scale, psf=psf, lens_galaxy=lens_galaxy, source_galaxy=source_galaxy,
    sub_grid_size=sub_grid_size, plot_ccd=plot_ccd)

###### SIE + SHEAR LENS 4 #######

data_name = 'sie_shear_x1_source_4'
lens_galaxy = g.Galaxy(mass=mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.3, axis_ratio=0.9, phi=100.0),
                       shear=mp.ExternalShear(magnitude=0.15, phi=170.0))

source_galaxy = g.Galaxy(light=lp.EllipticalSersic(centre=(0.5, -0.1), axis_ratio=0.8, phi=60.0,
                                                   intensity=0.2, effective_radius=0.5, sersic_index=3.0))

tools.from_data_lens_and_source_galaxy_simulate_and_output_ccd_data(data_path=data_path, data_name=data_name,
    ccd_shape=ccd_shape, pixel_scale=pixel_scale, psf=psf, lens_galaxy=lens_galaxy, source_galaxy=source_galaxy,
    sub_grid_size=sub_grid_size, plot_ccd=plot_ccd)