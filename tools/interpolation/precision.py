import autofit as af
from autolens.data import ccd
from autolens.data.array import grids
from autolens.model.profiles import mass_profiles as mp
from autolens.lens import lens_data as ld
from autolens.data.array import mask as msk

from autolens.plotters import array_plotters

import numpy as np

import os

# Setup the path to the workspace, using a relative directory name.
workspace_path = '{}/../'.format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(config_path=workspace_path + 'config', output_path=workspace_path + 'output')

# Although we could test the deflection angles without using an image (e.g. by just making a grid), we have chosen to
# set this test up using an image and mask. This gives run-time numbers that can be easily related to an actual lens
# analysis
data_type = 'example'
data_name = 'lens_light_and_x1_source' # An example simulated image with lens light emission and a source galaxy.
pixel_scale = 0.1

sub_grid_size = 2
inner_radius_arcsec = 0.0
outer_radius_arcsec = 3.0

print('sub grid size = ' + str(sub_grid_size))
print('annular inner mask radius = ' + str(inner_radius_arcsec) + '\n')
print('annular outer mask radius = ' + str(outer_radius_arcsec) + '\n')

print()

# Create the path where the data will be loaded from, which in this case is
# '/workspace/data/example/lens_light_and_x1_source/'
data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=['data', data_type, data_name])

ccd_data = ccd.load_ccd_data_from_fits(image_path=data_path + '/image.fits',
                                       psf_path=data_path + '/psf.fits',
                                       noise_map_path=data_path + '/noise_map.fits',
                                       pixel_scale=pixel_scale)

mask = msk.Mask.circular_annular(shape=ccd_data.shape, pixel_scale=ccd_data.pixel_scale,
                                 inner_radius_arcsec=inner_radius_arcsec, outer_radius_arcsec=outer_radius_arcsec)

lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, sub_grid_size=sub_grid_size)

print('Number of sub-grid points = ' + str(lens_data.grid_stack.sub.shape[0]) + '\n')

lens_data = ld.LensData(ccd_data=ccd_data, mask=mask, sub_grid_size=sub_grid_size)

interpolator = grids.Interpolator.from_mask_grid_and_interp_pixel_scales(
    mask=lens_data.mask_2d, grid=lens_data.grid_stack.sub, interp_pixel_scale=0.1)

print('Number of interpolation points = ' + str(interpolator.interp_grid.shape[0]) + '\n')

mass_profile = mp.EllipticalIsothermal(centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=0.5)

interp_deflections = mass_profile.deflections_from_grid(grid=interpolator.interp_grid)
deflections = np.zeros((lens_data.grid_stack.sub.shape[0], 2))
deflections[:,0] = interpolator.interpolated_values_from_values(values=interp_deflections[:,0])
deflections[:,1] = interpolator.interpolated_values_from_values(values=interp_deflections[:,1])

true_deflections = mass_profile.deflections_from_grid(grid=lens_data.grid_stack.sub)

difference_y = deflections[:,0] - true_deflections[:, 0]
difference_x = deflections[:,1] - true_deflections[:, 1]

print("interpolation y error: ", np.mean(difference_y))
print("interpolation y uncertainty: ", np.std(difference_y))
print("interpolation y max error: ", np.max(difference_y))
print("interpolation x error: ", np.mean(difference_x))
print("interpolation x uncertainty: ", np.std(difference_x))
print("interpolation x max error: ", np.max(difference_x))

difference_y_2d = lens_data.grid_stack.sub.scaled_array_2d_with_sub_dimensions_from_sub_array_1d_and_sub_grid_size(
    sub_array_1d=difference_y)
difference_x_2d = lens_data.grid_stack.sub.scaled_array_2d_with_sub_dimensions_from_sub_array_1d_and_sub_grid_size(
    sub_array_1d=difference_x)

array_plotters.plot_array(array=difference_y_2d)
array_plotters.plot_array(array=difference_x_2d)