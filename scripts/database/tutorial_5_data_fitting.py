"""
Database 5: Data Fitting
========================

In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to perform
fits to the data.

It is here the use of generators is absolutely essential. We are going to manipulating datasets which use a lot of
memory.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

"""
__Database File__

First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator.from_database(path.join("output", "database.sqlite"))

"""
The masks we used to fit the lenses is accessible via the aggregator.
"""
mask_gen = agg.values("mask")
print([mask for mask in mask_gen])

"""
The info dictionary we passed is also available.
"""
print("Info:")
info_gen = agg.values("info")
print([info for info in info_gen])

"""
__Dataset via List__

We can also use the aggregator to load the dataset of every lens our search fitted. 

The individual masked `data`, `noise_map` and `psf` are stored in the database, as opposed to the `Imaging` object, 
which saves of hard-disk space used. Thus, we need to create the `Imaging` object ourselves to inspect it. 
"""
data_gen = agg.values(name="data")
noise_map_gen = agg.values(name="noise_map")
psf_gen = agg.values(name="psf")
settings_imaging_gen = agg.values(name="settings_dataset")

for (data, noise_map, psf, settings_imaging) in zip(
    data_gen, noise_map_gen, psf_gen, settings_imaging_gen
):

    imaging = al.Imaging(
        image=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_imaging,
        setup_convolver=True,
    )

    imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
    imaging_plotter.subplot_imaging()

"""
__Dataset via Generators__

We should be doing this using a generator, as shown below.
"""


def make_imaging_gen(fit,):

    data = fit.value(name="data")
    noise_map = fit.value(name="noise_map")
    psf = fit.value(name="psf")
    settings_imaging = fit.value(name="settings_dataset")

    imaging = al.Imaging(
        image=data,
        noise_map=noise_map,
        psf=psf,
        settings=settings_imaging,
        setup_convolver=True,
    )

    imaging.apply_settings(settings=settings_imaging)

    return imaging


imaging_gen = agg.map(func=make_imaging_gen)

for imaging in imaging_gen:

    imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
    imaging_plotter.subplot_imaging()


"""
__Performing a Fit__

We now have access to the `Imaging` data we used to perform a model-fit, and the results of that model-fit in the form
of a `Samples` object. 

We can therefore use the database to create a `FitImaging` of the maximum log-likelihood model of every model to its
corresponding dataset, via the following generator:
"""


def make_fit_generator(fit):

    imaging = make_imaging_gen(fit=fit)

    tracer = al.Tracer.from_galaxies(galaxies=fit.instance.galaxies)

    return al.FitImaging(imaging=imaging, tracer=tracer)


fit_gen = agg.map(func=make_fit_generator)

for fit in fit_gen:

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_imaging_plotter.subplot_fit_imaging()

"""
The `AnalysisImaging` object has `settings_pixelization` and `settings_inversion` attributes, which customizes how 
these are used to fit the data. The generator above uses the `settings` of the object that were used by the model-fit. 

These settings objected are contained in the database and can therefore also be passed to the `FitImaging`.
"""


def make_fit_generator(fit):

    imaging = make_imaging_gen(fit=fit)

    settings_pixelization = fit.value(name="settings_pixelization")
    settings_inversion = fit.value(name="settings_inversion")

    tracer = al.Tracer.from_galaxies(galaxies=fit.instance.galaxies)

    return al.FitImaging(
        imaging=imaging,
        tracer=tracer,
        settings_pixelization=settings_pixelization,
        settings_inversion=settings_inversion,
    )


fit_gen = agg.map(func=make_fit_generator)

for fit in fit_gen:

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_imaging_plotter.subplot_fit_imaging()

"""
__Convenience Methods__

The PyAutoLens aggregator module also provides convenience methods for setting up objects *within* a generator. Below, 
we make the `Imaging` and `Tracer` using these methods, which perform the same functions as the generator above, 
including the settings.
"""


def plot_fit(agg_obj):

    imaging = al.agg.imaging_via_database_from(fit=agg_obj)
    tracer = al.agg.tracer_via_database_from(fit=agg_obj)

    return al.FitImaging(imaging=imaging, tracer=tracer)


fit_gen = agg.map(func=make_fit_generator)

for fit in fit_gen:
    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_imaging_plotter.subplot_fit_imaging()

"""
Of course, we also provide a convenience method to directly make the Imaging and FitImaging generators!
"""
imaging_gen = al.agg.Imaging(aggregator=agg)

for imaging in imaging_gen:
    print(imaging)

fit_gen = al.agg.FitImaging(aggregator=agg)

for fit in fit_gen:
    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_imaging_plotter.subplot_fit_imaging()

"""
This convenience method goes one step further. By default, it uses the `SettingsImaging`, `SettingsPixelization`
and `SettingsInversion` used by the analysis. 

However, we can change these settings such that the model-fit is performed differently. For example, what if I wanted 
to see how the fit looks where the `Grid2D`'s `sub_size` is 4 (instead of the value of 2 that was used)? Or where 
the `Pixelization` didn`t use a border? You can do this by passing settings objects to the method, which overwrite 
the ones used by the analysis.
"""
settings_imaging = al.SettingsImaging(sub_size=4)

imaging_gen = al.agg.Imaging(aggregator=agg, settings_imaging=settings_imaging)

settings_pixelization = al.SettingsPixelization(use_border=False)

fit_gen = al.agg.FitImaging(
    aggregator=agg,
    settings_imaging=settings_imaging,
    settings_pixelization=settings_pixelization,
)

for fit in fit_gen:
    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit)
    fit_imaging_plotter.subplot_fit_imaging()

"""
__Visualization Customization__

The benefit of inspecting fits using the aggregator, rather than the files outputs to the hard-disk, is that we can 
customize the plots using the PyAutoLens mat_plot_2d.

Below, we create a new function to apply as a generator to do this. However, we use a convenience method available 
in the PyAutoLens aggregator package to set up the fit.
"""
fit_gen = al.agg.FitImaging(aggregator=agg)

for fit in fit_gen:

    mat_plot_2d = aplt.MatPlot2D(
        figure=aplt.Figure(figsize=(12, 12)),
        title=aplt.Title(label="Custom Image", fontsize=24),
        yticks=aplt.YTicks(fontsize=24),
        xticks=aplt.XTicks(fontsize=24),
        cmap=aplt.Cmap(norm="log", vmax=1.0, vmin=1.0),
        colorbar_tickparams=aplt.ColorbarTickParams(labelsize=20),
        units=aplt.Units(in_kpc=True),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
    fit_imaging_plotter.figures_2d(normalized_residual_map=True)

"""
Making this plot for a paper? You can output it to hard disk.
"""
fit_gen = al.agg.FitImaging(aggregator=agg)

for fit in fit_gen:

    mat_plot_2d = aplt.MatPlot2D(
        title=aplt.Title(label="Hey"),
        output=aplt.Output(
            path=path.join("output", "path", "of", "file"),
            filename="publication",
            format="png",
        ),
    )

    fit_imaging_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
    fit_imaging_plotter.figures_2d(normalized_residual_map=True)

"""
Finished.
"""
