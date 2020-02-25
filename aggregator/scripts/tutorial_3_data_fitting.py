import os

import autofit as af
import autolens as al
import autolens.plot as aplt

# In this tutorial, we use the aggregator to load models and data from a non-linear search and use them to reperform
# fits to the data.

# Below, we set up the aggregator as we did in the previous tutorial.
workspace_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = workspace_path + "output"
aggregator_results_path = output_path + "/aggregator_sample_beginner"

af.conf.instance = af.conf.Config(
    config_path=str(workspace_path + "/config"), output_path=str(output_path)
)

aggregator = af.Aggregator(directory=str(aggregator_results_path))

# Again, we create a list of the MultiNestOutputs of each phase.
pipeline_name = "pipeline__lens_sie__source_inversion"
phase_name = "phase_3__source_inversion"

outputs = aggregator.filter(phase=phase_name).output

# We can also use the aggregator to load the dataset of every lens our pipeline fitted. This returns the dataset
# as the "Imaging" objects we passed to the pipeline when we ran them.

datasets = aggregator.filter(phase=phase_name).dataset
print("Datasets:")
print(datasets, "\n")

# Lets plot each dataset's subplot.
[aplt.imaging.subplot_imaging(imaging=dataset) for dataset in datasets]

# We'll also need the masks we used to fit the lenses, which the aggregator also provides.
masks = aggregator.filter(phase=phase_name).mask
print("Masks:")
print(masks, "\n")

# Lets plot each dataset's again now with its mask.
[
    aplt.imaging.subplot_imaging(imaging=dataset, mask=mask)
    for dataset, mask in zip(datasets, masks)
]

# To reperform the fit of each most-likely lens model we'll need the masked imaging used by that phase.
masked_imagings = [
    al.masked.imaging(dataset=dataset, mask=mask)
    for dataset, mask in zip(datasets, masks)
]

# Okay, we're good to go! Lets use each most likely instance to create the most-likely tracer, and fit the masked
# imaging using this tracer for every lens.
instances = [out.most_likely_instance for out in outputs]

tracers = [
    al.Tracer.from_galaxies(galaxies=instance.galaxies) for instance in instances
]

fits = [
    al.fit(masked_dataset=masked_imaging, tracer=tracer)
    for masked_imaging, tracer in zip(masked_imagings, tracers)
]

[aplt.fit_imaging.subplot_fit_imaging(fit=fit) for fit in fits]

# The benefit of inspecting fits using the aggregator, rather than the files outputs to the hard-disk, is that
# we can customize the plots using the PyAutoLens plotters.

plotter = aplt.Plotter(
    figure=aplt.Figure(figsize=(12, 12)),
    labels=aplt.Labels(title="Custom Image", titlesize=24, ysize=24, xsize=24),
    ticks=aplt.Ticks(ysize=24, xsize=24),
    cmap=aplt.ColorMap(norm="log", norm_max=1.0, norm_min=1.0),
    cb=aplt.ColorBar(ticksize=20),
    units=aplt.Units(in_kpc=True),
)

[aplt.fit_imaging.normalized_residual_map(fit=fit, plotter=plotter) for fit in fits]

# Making this plot for a paper? You can output it to hard disk.

plotter = aplt.Plotter(
    output=aplt.Output(
        path=workspace_path + "/output/path/of/file/",
        filename="publication",
        format="png",
    )
)
