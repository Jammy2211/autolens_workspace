"""
GUI Preprocessing: Lens Light Centre
====================================

This tool allows one to input the lens light centre(s) of a strong lens(es) via a GUI, which can be used as a fixed
value in pipelines.

This GUI is adapted from the following code: https://gist.github.com/brikeats/4f63f867fd8ea0f196c78e9b835150ab
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt
from matplotlib import pyplot as plt
import numpy as np

"""
Setup the path the datasets we'll use to illustrate preprocessing, which is the 
folder `dataset/imaging/with_lens_light/light_sersic__mass_sie__source_sersic`.
"""
dataset_path = path.join("..", "sdssj1152p3312")

"""
If you use this tool for your own dataset, you *must* double check this pixel scale is correct!
"""
pixel_scales = 0.03

"""
Load the image which we will use to mark the lens light centre.
"""
image = al.Array2D.from_fits(
    file_path=path.join(dataset_path, "f160w_image.fits"),
    hdu=0,
    pixel_scales=pixel_scales,
)
image_2d = image.native

"""
When you click on a pixel to mark a position, the search box looks around this click and finds the pixel with
the highest flux to mark the position.

The `search_box_size` is the number of pixels around your click this search takes place.
"""
search_box_size = 5

"""
The `norm` object created below customizes the minimum and maximum values of the colormap that is plotted, which can be
useful for datasets with a high dynamic range.
"""
cmap = aplt.Cmap(norm="linear", vmin=0.0, vmax=0.05)

norm = cmap.norm_from_array(array=None)

"""
This code is a bit messy, but sets the image up as a matplotlib figure which one can double click on to mark the
positions on an image.
"""
light_centres = []


def onclick(event):
    if event.dblclick:

        y_arcsec = np.rint(event.ydata / pixel_scales) * pixel_scales
        x_arcsec = np.rint(event.xdata / pixel_scales) * pixel_scales

        (y_pixels, x_pixels) = image.mask.pixel_coordinates_2d_from(
            scaled_coordinates_2d=(y_arcsec, x_arcsec)
        )

        flux = -np.inf

        for y in range(y_pixels - search_box_size, y_pixels + search_box_size):
            for x in range(x_pixels - search_box_size, x_pixels + search_box_size):
                flux_new = image_2d[y, x]
                if flux_new > flux:
                    flux = flux_new
                    y_pixels_max = y
                    x_pixels_max = x

        grid_arcsec = image.mask.grid_scaled_from_grid_pixels_1d(
            grid_pixels_1d=al.Grid2D.manual_native(
                grid=[[[y_pixels_max + 0.5, x_pixels_max + 0.5]]],
                pixel_scales=pixel_scales,
            )
        )
        y_arcsec = grid_arcsec[0, 0]
        x_arcsec = grid_arcsec[0, 1]

        print("clicked on:", y_pixels, x_pixels)
        print("Max flux pixel:", y_pixels_max, x_pixels_max)
        print("Arc-sec Coordinate", y_arcsec, x_arcsec)

        light_centres.append((y_arcsec, x_arcsec))


n_y, n_x = image.shape_native
hw = int(n_x / 2) * pixel_scales
ext = [-hw, hw, -hw, hw]
fig = plt.figure(figsize=(14, 14))
plt.imshow(image.native, cmap="jet", extent=ext, norm=norm)
plt.colorbar()
cid = fig.canvas.mpl_connect("button_press_event", onclick)
plt.show()
fig.canvas.mpl_disconnect(cid)
plt.close(fig)

light_centres = al.Grid2DIrregular(grid=light_centres)

"""
Now lets plot the image and lens light centre, so we can check that the centre overlaps the brightest pixel in the
lens light.
"""
visuals_2d = aplt.Visuals2D(light_profile_centres=light_centres)
aplt.Array2DPlotter(array=image, visuals_2d=visuals_2d)

"""
Now we`re happy with the lens light centre(s), lets output them to the dataset folder of the lens, so that we can 
load them from a.json file in our pipelines!
"""
try:
    light_centres.output_to_json(
        file_path=path.join(dataset_path, "light_centres.json"), overwrite=True
    )
except AttributeError:
    pass
