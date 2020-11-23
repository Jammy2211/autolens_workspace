from os import path
import autolens as al
import autolens.plot as aplt
from matplotlib import pyplot as plt
import numpy as np

"""
This tool allows one to input the lens light centre(s) of a strong lens(es), which can be used as a fixed value in
pipelines. 

First, we set up the dataset we want to mark the positions of.
"""

dataset_name = "mass_sie__source_sersic"

"""
The path where the dataset will be loaded from, which in this case is:

 `/autolens_workspace/dataset/imaging/no_lens_light/mass_sie__source_sersic`
"""

dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

"""If you use this tool for your own dataset, you *must* double check this pixel scale is correct!"""

pixel_scales = 0.1

"""
When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
the highest flux to mark the position.

The `search_box_size` is the number of pixels around your click this search takes place.
"""

search_box_size = 5

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=pixel_scales,
)
image_2d = imaging.image.in_2d

"""
For lenses with bright lens light emission, it can be difficult to get the source light to show. The normalization
below uses a log-scale with a capped maximum, which better contrasts the lens and source emission.
"""

cmap = aplt.ColorMap(
    norm="linear",
    norm_min=1.0e-4,
    norm_max=np.max(imaging.image),
    #   linthresh=0.05,
    #   linscale=0.1,
)

norm = cmap.norm_from_array(array=None)

positions = []

"""
This code is a bit messy, but sets the image up as a matplotlib figure which one can double click on to mark the
positions on an image.
"""


def onclick(event):
    if event.dblclick:

        # y_arcsec = np.rint(event.ydata / pixel_scales) * pixel_scales
        # x_arcsec = np.rint(event.xdata / pixel_scales) * pixel_scales
        #
        # (
        #     y_pixels,
        #     x_pixels,
        # ) = image_2d.geometry.pixel_coordinates_2d_from(
        #     scaled_coordinates_2d=(y_arcsec, x_arcsec)
        # )

        y_pixels = event.ydata
        x_pixels = event.xdata

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
            grid_pixels_1d=al.Grid.manual_2d(
                grid=[[[y_pixels_max + 0.5, x_pixels_max + 0.5]]],
                pixel_scales=pixel_scales,
            )
        )
        y_arcsec = grid_arcsec[0, 0]
        x_arcsec = grid_arcsec[0, 1]

        print("clicked on:", y_pixels, x_pixels)
        print("Max flux pixel:", y_pixels_max, x_pixels_max)
        print("Arc-sec Coordinate", y_arcsec, x_arcsec)

        positions.append((y_arcsec, x_arcsec))


n_y, n_x = imaging.image.shape_2d
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
plt.imshow(imaging.image.in_2d, cmap="jet", extent=ext, norm=norm)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

positions = al.GridIrregularGrouped(grid=positions)

"""Now lets plot the image and positions, so we can check that the positions overlap different regions of the source."""

aplt.Array(array=imaging.image, positions=positions)

"""
Now we`re happy with the positions, lets output them to the dataset folder of the lens, so that we can load them from a
.dat file in our pipelines!
"""

positions.output_to_file(
    file_path=path.join(dataset_path, "positions.dat"), overwrite=True
)
