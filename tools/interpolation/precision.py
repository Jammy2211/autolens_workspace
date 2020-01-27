import autofit as af
import autolens as al
import autolens.plot as aplt
import numpy as np

import os

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=workspace_path + "config", output_path=workspace_path + "output"
)

# Although we could test_autoarray the deflection angles without using an image (e.g. by just making a grid), we have chosen to
# set this test_autoarray up using an image and mask. This gives run-time numbers that can be easily related to an actual lens
# analysis
dataset_label = "imaging"
dataset_name = (
    "lens_sie__source_sersic"
)  # An example simulated image with lens light emission and a source galaxy.
pixel_scales = 0.1

sub_size = 2
inner_radius = 0.0
outer_radius = 3.0

print("sub grid size = " + str(sub_size))
print("annular inner mask radius = " + str(inner_radius) + "\n")
print("annular outer mask radius = " + str(outer_radius) + "\n")

print()

# Create the path where the dataset will be loaded from, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

imaging = al.imaging.from_fits(
    image_path=dataset_path + "/image.fits",
    psf_path=dataset_path + "/psf.fits",
    noise_map_path=dataset_path + "/noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.mask.circular_annular(
    shape_2d=imaging.shape_2d,
    pixel_scales=imaging.pixel_scales,
    inner_radius=inner_radius,
    outer_radius=outer_radius,
)

masked_imaging = al.masked.imaging(imaging=imaging, mask=mask, sub_size=sub_size)

print("Number of sub-grid points = " + str(masked_imaging.grid.shape[0]) + "\n")

masked_imaging = al.masked.imaging(imaging=imaging, mask=mask, sub_size=sub_size)

interpolator = al.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
    mask=masked_imaging.mask_2d,
    grid=masked_imaging.grid,
    pixel_scale_interpolation_grid=0.1,
)

print(
    "Number of interpolation points = " + str(interpolator.interp_grid.shape[0]) + "\n"
)

mass_profile = al.mp.EllipticalIsothermal(
    centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, einstein_radius=0.5
)

interp_deflections = mass_profile.deflections_from_grid(grid=interpolator.interp_grid)
deflections = np.zeros((masked_imaging.grid.shape[0], 2))
deflections[:, 0] = interpolator.interpolated_values_from_values(
    values=interp_deflections[:, 0]
)
deflections[:, 1] = interpolator.interpolated_values_from_values(
    values=interp_deflections[:, 1]
)

true_deflections = mass_profile.deflections_from_grid(grid=masked_imaging.grid)

difference_y = deflections[:, 0] - true_deflections[:, 0]
difference_x = deflections[:, 1] - true_deflections[:, 1]

print("interpolation y error: ", np.mean(difference_y))
print("interpolation y uncertainty: ", np.std(difference_y))
print("interpolation y max error: ", np.max(difference_y))
print("interpolation x error: ", np.mean(difference_x))
print("interpolation x uncertainty: ", np.std(difference_x))
print("interpolation x max error: ", np.max(difference_x))

difference_y_2d = masked_imaging.grid.array_stored_1d_from_sub_array_1d(
    sub_array_1d=difference_y
)
difference_x_2d = masked_imaging.grid.array_stored_1d_from_sub_array_1d(
    sub_array_1d=difference_x
)

aplt.array(array=difference_y_2d)
aplt.array(array=difference_x_2d)
