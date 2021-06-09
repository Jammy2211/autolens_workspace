"""
Plots: InterferometerPlotter
============================

This example illustrates how to plot an `Interferometer` dataset using an `InterferometerPlotter`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

First, lets load example interferometer of of a strong lens as an `Interferometer` object.
"""
dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "interferometer", dataset_name)

real_space_mask = al.Mask2D.circular(
    shape_native=(200, 200), pixel_scales=0.05, radius=3.0
)

interferometer = al.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
__Figures__

We now pass the interferometer to an `InterferometerPlotter` and call various `figure_*` methods to plot different 
attributes.
"""
interferometer_plotter = aplt.InterferometerPlotter(interferometer=interferometer)
interferometer_plotter.figures_2d(
    visibilities=True,
    noise_map=True,
    u_wavelengths=True,
    v_wavelengths=True,
    uv_wavelengths=True,
    amplitudes_vs_uv_distances=True,
    phases_vs_uv_distances=True,
)

"""
The dirty images of the interferometer dataset can also be plotted, which use the transformer of the interferometer 
to map the visibilities, noise-map or other quantity to a real-space image.
"""
interferometer_plotter.figures_2d(
    dirty_image=True,
    dirty_noise_map=True,
    dirty_signal_to_noise_map=True,
    dirty_inverse_noise_map=True,
)

"""
__Subplots__

The `InterferometerPlotter` may also plot a subplot of all of these attributes.
"""
interferometer_plotter.subplot_interferometer()
interferometer_plotter.subplot_dirty_images()

"""
__Include__

The `Interferometer` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include_2d = aplt.Include2D()

interferometer_plotter = aplt.InterferometerPlotter(
    interferometer=interferometer, include_2d=include_2d
)
interferometer_plotter.subplot_interferometer()

"""
Finish.
"""
