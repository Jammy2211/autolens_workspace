"""
Plots: FitImagingPlotter
========================

This example illustrates how to plot an `FitImaging` object using an `FitImagingPlotter`.
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

First, lets load example imaging of of a strong lens as an `Imaging` object.
"""
dataset_name = "light_sersic__mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "with_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Fit__

We now mask the data and fit it with a `Tracer` to create a `FitImaging` object.
"""
mask = al.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.EllIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.05, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllSersic(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=0.3,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(imaging=imaging, tracer=tracer)

"""
__Figures__

We now pass the FitImaging to an `FitImagingPlotter` and call various `figure_*` methods to plot different attributes.
"""
fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
fit_imaging_plotter.figures_2d(
    image=True,
    noise_map=True,
    signal_to_noise_map=True,
    model_image=True,
    residual_map=True,
    normalized_residual_map=True,
    chi_squared_map=True,
)

"""
It can plot of the model image of an input plane.
"""
fit_imaging_plotter.figures_2d_of_planes(model_image=True, plane_index=0)
fit_imaging_plotter.figures_2d_of_planes(model_image=True, plane_index=1)

"""
It can plot the image of a plane with all other model images subtracted.
"""
fit_imaging_plotter.figures_2d_of_planes(subtracted_image=True, plane_index=0)
fit_imaging_plotter.figures_2d_of_planes(subtracted_image=True, plane_index=1)

"""
It can also plot the plane-image of a plane, that is what the source galaxy looks like without lensing (e.g.
for `plane_index=1` this is the source-plane image)
"""
fit_imaging_plotter.figures_2d_of_planes(plane_index=0)
fit_imaging_plotter.figures_2d_of_planes(plane_index=1)

"""
__Subplots__

The `FitImagingPlotter` may also plot a subplot of these attributes.
"""
fit_imaging_plotter.subplot_fit_imaging()
fit_imaging_plotter.subplot_of_planes(plane_index=1)

"""`
__Include__

`FitImaging` contains the following attributes which can be plotted automatically via the `Include2D` object.
"""
include_2d = aplt.Include2D(
    origin=True,
    mask=True,
    border=True,
    light_profile_centres=True,
    mass_profile_centres=True,
    critical_curves=True,
    caustics=True,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_plotter.subplot_fit_imaging()
fit_plotter.subplot_of_planes(plane_index=0)
fit_plotter.subplot_of_planes(plane_index=1)

"""
__Inversion__

We can also plot a `FitImaging` which uses an `Inversion`.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    pixelization=al.pix.VoronoiMagnification(shape=(25, 25)),
    regularization=al.reg.Constant(coefficient=1.0),
)

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(imaging=imaging, tracer=tracer)

"""
__Include__

The `plane_image_from_plane` method now plots the the reconstructed source on the Voronoi pixel-grid. It can use the
`Include2D` object to plot the `Mapper`'s specific structures like the image and source plane pixelization grids.
"""
include_2d = aplt.Include2D(
    mapper_data_pixelization_grid=True, mapper_source_grid_slim=True
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_plotter.figures_2d_of_planes(plane_image=True, plane_index=1)

"""
In fact, via the `FitImagingPlotter` we can plot the `reconstruction` with caustics and a border, which are extracted 
from the `Tracer` of the `FitImaging`. 

To do this with an `InversionPlotter` we would have had to manually pass these attributes via the `Visuals2D` object.
"""
include_2d = aplt.Include2D(
    border=True,
    caustics=True,
    mapper_data_pixelization_grid=True,
    mapper_source_grid_slim=True,
)

fit_plotter = aplt.FitImagingPlotter(fit=fit, include_2d=include_2d)
fit_plotter.figures_2d_of_planes(plane_image=True, plane_index=1)

"""
__Inversion Plotter__

We can even extract an `InversionPlotter` from the `FitImagingPlotter` and use it to plot all of its usual methods, 
which will now include the caustic and border.
"""
inversion_plotter = fit_plotter.inversion_plotter_of_plane(plane_index=1)
inversion_plotter.figures_2d(reconstruction=True, regularization_weight_list=True)

"""
Finish.
"""
