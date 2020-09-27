# %%
"""
__Aggregator 3: Data Fitting__

In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to reperform
fits to the data.

It is here the use of generators is absolutely essential. We are going to manipulating datasets which use a lot of
memory.
"""

# %%
from autoconf import conf
import autofit as af
import autolens as al
import autolens.plot as aplt

# %%
"""
Below, we set up the aggregator as we did in the previous tutorial.
"""

# %%
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"
output_path = f"{workspace_path}/output"
agg_results_path = f"{output_path}/aggregator"

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=output_path
)

agg = af.Aggregator(directory=str(agg_results_path))

# %%
"""
Again, we create a list of the Samples of each phase.
"""

# %%
phase_name = "phase__aggregator"
agg_filter = agg.filter(agg.phase == phase_name)

# %%
"""
We can also use the aggregator to load the dataset of every lens our `Phase` fitted. This generator returns the 
dataset as the `Imaging` objects we passed to the phase when we ran them.
"""

# %%
dataset_gen = agg_filter.values("dataset")

print("Datasets:")
print(dataset_gen, "\n")
print(list(dataset_gen)[0].image)

for dataset in agg_filter.values("dataset"):

    aplt.Imaging.subplot_imaging(imaging=dataset)

# %%
"""
The name of the dataset we assigned when we ran the phase is also available, which helps us to label the lenses 
on plots.
"""

# %%
print("Dataset Names:")
dataset_gen = agg_filter.values("dataset")
print([dataset.name for dataset in dataset_gen])

# %%
"""
The info dictionary we passed is also available.
"""

# %%
print("Info:")
info_gen = agg_filter.values("info")
print([info for info in info_gen])

# %%
"""
we'll also need the masks we used to fit the lenses, which the aggregator also provides.
"""

# %%
mask_gen = agg_filter.values("mask")
print("Masks:")
print(list(mask_gen), "\n")

# %%
"""
Lets plot each dataset again now with its mask, using generators.
"""

# %%

dataset_gen = agg_filter.values("dataset")
mask_gen = agg_filter.values("mask")

for dataset, mask in zip(dataset_gen, mask_gen):
    aplt.Imaging.subplot_imaging(imaging=dataset, mask=mask)

# %%
"""
To reperform the fit of each maximum log likelihood lens model we can use the following generator.
"""

# %%
def make_fit_generator(agg_obj):

    output = agg_obj.samples
    dataset = agg_obj.dataset
    mask = agg_obj.mask

    masked_imaging = al.MaskedImaging(imaging=dataset, mask=mask)

    tracer = al.Tracer.from_galaxies(
        galaxies=output.max_log_likelihood_instance.galaxies
    )

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


fit_gen = agg_filter.map(func=make_fit_generator)

for fit in fit_gen:

    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
There`s a problem though, the MaskedImaging object is made with a custom phase input. For example, it receive a 
`grid_class` defining which grid it uses to fit the data. This isn't included in the generator above. Thats bad!

The output has a meta_dataset attribute containing all the information on how the MaskedImaging was created for the
actualy phase.
"""

# %%
def make_fit_generator(agg_obj):

    output = agg_obj.samples
    dataset = agg_obj.dataset
    mask = agg_obj.mask

    """This corresponds to `SettingsPhaseImaging` used un the runner script."""

    settings = agg_obj.settings

    masked_imaging = al.MaskedImaging(
        imaging=dataset, mask=mask, settings=settings.settings_masked_imaging
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=output.max_log_likelihood_instance.galaxies
    )

    return al.FitImaging(
        masked_imaging=masked_imaging,
        tracer=tracer,
        settings_pixelization=settings.settings_pixelization,
        settings_inversion=settings.settings_inversion,
    )


fit_gen = agg_filter.map(func=make_fit_generator)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
Thats a lot of input parameters! The good news is this means in the aggregator we can customize exactly how the 
MaskedImaging is set up. The bad news is this requires a lot of lines of code, which is prone to typos and errors. 

If you are writing customized generator functions, the PyAutoLens aggregator module also provides convenience methods
for setting up objects *within* a generator. Below, we make the MaskedImaging and Tracer using these methods, which
perform the same functions as the generator above, including the SettingsPhaseImaging.
"""

# %%
def plot_fit(agg_obj):

    masked_imaging = al.agg.masked_imaging_from_agg_obj(agg_obj=agg_obj)

    tracer = al.agg.tracer_from_agg_obj(agg_obj=agg_obj)

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


fit_gen = agg_filter.map(func=make_fit_generator)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
Of course, we also provide a convenience method to directly make the MaskedImaging and FitImaging generators!
"""

# %%
masked_imaging_gen = al.agg.MaskedImaging(aggregator=agg_filter)

for masked_imaging in masked_imaging_gen:
    print(masked_imaging.name)

fit_gen = al.agg.FitImaging(aggregator=agg_filter)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
This convenience method goes one step further. By default, it uses the `SettingsMaskedImaging`, _SettingsPixelization_
and `SettingsInversion` used by the analysis. 

However, we may want to change this. For example, what if I was curious and wanted to see the fit but where I used
a `Grid` with a *sub_size* of 4? Or where the `Pixelization` didn`t use a border? You can do this by passing the
method a new `Settings` object which overwrites the one used by the analysis.
"""

# %%

settings_masked_imaging = al.SettingsMaskedImaging(sub_size=4)

masked_imaging_gen = al.agg.MaskedImaging(
    aggregator=agg_filter, settings_masked_imaging=settings_masked_imaging
)

settings_pixelization = al.SettingsPixelization(use_border=False)

fit_gen = al.agg.FitImaging(
    aggregator=agg_filter,
    settings_masked_imaging=settings_masked_imaging,
    settings_pixelization=settings_pixelization,
)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
The benefit of inspecting fits using the aggregator, rather than the files outputs to the hard-disk, is that we can 
customize the plots using the PyAutoLens plotters.

Below, we create a new function to apply as a generator to do this. However, we use a convenience method available 
in the PyAutoLens aggregator package to set up the fit.
"""

# %%
fit_gen = al.agg.FitImaging(aggregator=agg_filter)

for fit in fit_gen:

    plotter = aplt.Plotter(
        figure=aplt.Figure(figsize=(12, 12)),
        labels=aplt.Labels(title="Custom Image", titlesize=24, ysize=24, xsize=24),
        ticks=aplt.Ticks(ysize=24, xsize=24),
        cmap=aplt.ColorMap(norm="log", norm_max=1.0, norm_min=1.0),
        cb=aplt.ColorBar(ticksize=20),
        units=aplt.Units(in_kpc=True),
    )

    aplt.FitImaging.normalized_residual_map(fit=fit, plotter=plotter)

# %%
"""
Making this plot for a paper? You can output it to hard disk.
"""

# %%
fit_gen = al.agg.FitImaging(aggregator=agg_filter)

for fit in fit_gen:

    plotter = aplt.Plotter(
        labels=aplt.Labels(title="Hey"),
        output=aplt.Output(
            path=f"{workspace_path}/output/path/of/file/",
            filename="publication",
            format="png",
        ),
    )

    aplt.FitImaging.normalized_residual_map(fit=fit, plotter=plotter)
