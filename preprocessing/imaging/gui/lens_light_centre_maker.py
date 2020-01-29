import autolens as al
import autolens.plot as aplt
import autofit as af
import os
from matplotlib import pyplot as plt
import numpy as np

# This tool allows one to input the lens light centre(s) of a strong lens(es), which can be used as a fixed value in
# pipelines.

# Setup the path to the autolens_workspace, using a relative directory name.
workspace_path = "{}/../../../".format(os.path.dirname(os.path.realpath(__file__)))

# The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the positions are stored in e.g,
# the positions will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/positions.dat'.
dataset_label = "imaging"
dataset_name = "lens_sersic_sie__source_sersic"

# Create the path where the mask will be output, which in this case is
# '/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
    path=workspace_path, folder_names=["dataset", dataset_label, dataset_name]
)

# If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
pixel_scales = 0.1

# When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
# the highest flux to mark the position.
search_box_size = 5

imaging = al.imaging.from_fits(
    image_path=dataset_path + "image.fits",
    psf_path=dataset_path + "psf.fits",
    noise_map_path=dataset_path + "noise_map.fits",
    pixel_scales=pixel_scales,
)
image_2d = imaging.image.in_2d

# For lenses with bright lens light emission, it can be difficult to get the source light to show. The normalization
# below uses a log-scale with a capped maximum, which better contrasts the lens and source emission.

lens_light_centres = []

# This code is a bit messy, but sets the image up as a matplotlib figure which one can double click on to mark the
# positions on an image.


def onclick(event):
    if event.dblclick:

        y_arcsec = np.rint(event.ydata / pixel_scales) * pixel_scales
        x_arcsec = np.rint(event.xdata / pixel_scales) * pixel_scales

        y_pixels, x_pixels = image_2d.geometry.pixel_coordinates_from_scaled_coordinates(
            scaled_coordinates=(y_arcsec, x_arcsec)
        )

        flux = -np.inf

        for y in range(y_pixels - search_box_size, y_pixels + search_box_size):
            for x in range(x_pixels - search_box_size, x_pixels + search_box_size):
                flux_new = image_2d[y, x]
                #      print(y, x, flux_new)
                if flux_new > flux:
                    flux = flux_new
                    y_pixels_max = y
                    x_pixels_max = x

        grid_arcsec = image_2d.geometry.grid_scaled_from_grid_pixels_1d(
            grid_pixels_1d=al.grid.manual_2d(
                grid=[[[y_pixels_max + 0.5, x_pixels_max + 0.5]]],
                pixel_scales=pixel_scales,
            )
        )
        y_arcsec = grid_arcsec[0, 0]
        x_arcsec = grid_arcsec[0, 1]

        print("clicked on:", y_pixels, x_pixels)
        print("Max flux pixel:", y_pixels_max, x_pixels_max)
        print("Arc-sec Coordinate", y_arcsec, x_arcsec)

        lens_light_centres.append([y_arcsec, x_arcsec])


n_y, n_x = imaging.image.shape_2d
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
plt.imshow(imaging.image.in_2d, cmap="jet", extent=ext)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

lens_light_centres = al.coordinates(coordinates=[lens_light_centres])

# Now lets plotters the image and positions, so we can check that the positions overlap different regions of the source.
aplt.array(array=imaging.image, light_profile_centres=lens_light_centres)

# Now we're happy with the positions, lets output them to the dataset folder of the lens, so that we can load them from a
# .dat file in our pipelines!
lens_light_centres.output_to_file(file_path=dataset_path + "lens_light_centres.dat")
