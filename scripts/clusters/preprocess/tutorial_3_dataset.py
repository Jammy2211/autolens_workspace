"""
Preprocess
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import autolens as al
import autolens.plot as aplt

dataset_path = path.join("..", "sdssj1152p3312")

image = al.Array2D.from_fits(file_path=path.join(dataset_path, "f160w_image.fits"), hdu=0, pixel_scales=0.03)

# visuals_2d = aplt.Visuals2D(positions=positions, light_profile_centres=lens_centre)
mat_plot_2d = aplt.MatPlot2D(
 #   axis=aplt.Axis(extent=[-1.5, 1.5, -2.5, 0.5]),
    cmap=aplt.Cmap(vmin=0.0, vmax=0.3)
)

array_plotter = aplt.Array2DPlotter(
    array=image.native, mat_plot_2d=mat_plot_2d
)
array_plotter.figure_2d()

# imaging = al.Imaging.from_fits(
#     image_path=path.join(dataset_path, "f160w_image.fits"),
#     psf_path=path.join(dataset_path, "psf.fits"),
#     noise_map_path=path.join(dataset_path, "noise_map.fits"),
#     pixel_scales=0.03,
# )
#
# imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
# imaging_plotter.subplot_imaging()