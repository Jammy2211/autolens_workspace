# %%
"""
__Aggregator 3: Data Fitting__

In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to reperform
fits to the data.

It is here the use of generators is absolutely essential. We are going to manipulating datasets which use a lot of
memory.
"""

# %%
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
agg_results_path = f"{output_path}/aggregator/beginner"

af.conf.instance = af.conf.Config(
    config_path=f"{workspace_path}/config", output_path=output_path
)

agg = af.Aggregator(directory=str(agg_results_path))

# %%
"""
Again, we create a list of the NestedSamplerSampless of each phase.
"""

# %%
pipeline_name = "pipeline__lens_sie__source_inversion"
phase_name = "phase_3__source_inversion"
agg_phase_3 = agg.filter(agg.phase == phase_name)

# %%
"""
We can also use the aggregator to load the dataset of every lens our pipeline fitted. This generator returns the 
dataset as the "Imaging" objects we passed to the pipeline when we ran them.
"""

# %%
dataset_gen = agg_phase_3.values("dataset")

print("Datasets:")
print(dataset_gen, "\n")
print(list(dataset_gen)[0].image)

for dataset in agg_phase_3.values("dataset"):

    aplt.Imaging.subplot_imaging(imaging=dataset)

# %%
"""
The name of the dataset we assigned when we ran the pipelines are also attatch, which will help us to label the lenses 
on our plots or inspect our results in terms of measurements not part of our lens modeling.
"""

# %%
print("Dataset Names:")
dataset_gen = agg_phase_3.values("dataset")
print([dataset.name for dataset in dataset_gen])

# %%
"""
The info dictionary we passed is also available.
"""

# %%
print("Info:")
info_gen = agg_phase_3.values("info")
print([info for info in info_gen])

# %%
"""
We'll also need the masks we used to fit the lenses, which the aggregator also provides.
"""

# %%
mask_gen = agg_phase_3.values("mask")
print("Masks:")
print(list(mask_gen), "\n")

# %%
"""
Lets plot each dataset again now with its mask, using generators.
"""

# %%

dataset_gen = agg_phase_3.values("dataset")
mask_gen = agg_phase_3.values("mask")

for dataset, mask in zip(dataset_gen, mask_gen):
    aplt.Imaging.subplot_imaging(imaging=dataset, mask=mask)

# %%
"""
To reperform the fit of each most-likely lens model we can use the following generator.
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


fit_gen = agg_phase_3.map(func=make_fit_generator)

for fit in fit_gen:

    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
There's a problem though - what if our MaskedImaging was made with a custom phase input? For example, we may have used
the option "inversion_uses_border" in our phase. The default input of this in MaskedImaging is False, so the generator
above would have set up this object incorrect. Thats bad!

The output has a meta_dataset attribute containing all the information on how the MaskedImaging was created for the
actualy phase.
"""

# %%
def make_fit_generator(agg_obj):

    output = agg_obj.samples
    dataset = agg_obj.dataset
    mask = agg_obj.mask
    meta_dataset = agg_obj.meta_dataset

    masked_imaging = al.MaskedImaging(
        imaging=dataset,
        mask=mask,
        psf_shape_2d=meta_dataset.psf_shape_2d,
        pixel_scale_interpolation_grid=meta_dataset.pixel_scale_interpolation_grid,
        inversion_pixel_limit=meta_dataset.inversion_pixel_limit,
        inversion_uses_border=meta_dataset.inversion_uses_border,
        positions_threshold=meta_dataset.positions_threshold,
    )

    tracer = al.Tracer.from_galaxies(
        galaxies=output.max_log_likelihood_instance.galaxies
    )

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


fit_gen = agg_phase_3.map(func=make_fit_generator)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
Thats a lot of input parameters! The good news is this means in the aggregator we can customize exactly how the 
MaskedImaging is set up - we could check to see what happens if we switch the inversion border off, for example. The 
bad news is this requires a lot of lines of code, which is prone to typos and errors. 

If you are writing customized generator functions, the PyAutoLens aggregator module also provides convenience methods
for setting up objects *within* a generator. Below, we make the MaskedImaging and Tracer using these methods, which
perform the same functions as the generator above.
"""

# %%
def plot_fit(agg_obj):

    masked_imaging = al.agg.masked_imaging_from_agg_obj(agg_obj=agg_obj)

    tracer = al.agg.tracer_from_agg_obj(agg_obj=agg_obj)

    return al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)


fit_gen = agg_phase_3.map(func=make_fit_generator)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""
Of course, we also provide a convenience method to directly make the MaskedImaging and FitImaging generators!
"""

# %%
masked_imaging_gen = al.agg.MaskedImaging(aggregator=agg_phase_3)

for masked_imaging in masked_imaging_gen:
    print(masked_imaging.name)

fit_gen = al.agg.FitImaging(aggregator=agg_phase_3)

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
fit_gen = al.agg.FitImaging(aggregator=agg_phase_3)

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
fit_gen = al.agg.FitImaging(aggregator=agg_phase_3)

for fit in fit_gen:

    plotter = aplt.Plotter(
        output=aplt.Output(
            path=f"{workspace_path}//output/path/of/file/",
            filename="publication",
            format="png",
        )
    )

    aplt.FitImaging.normalized_residual_map(fit=fit, plotter=plotter)
