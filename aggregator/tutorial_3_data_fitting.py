from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

# This tutorial builds on tutorial_1 of the aggregator autolens_workspace. Here, we will use the aggregator to load
# models from a non-linear search and use them to visualize and interpret results.

# Below, we set up the aggregator as we did in the previous tutorial.
workspace_path = Path(__file__).parent.parent
output_path = workspace_path / "output"
aggregator_results_path = output_path / "aggregator_sample_beginner"

af.conf.instance = af.conf.Config(
    config_path=str(workspace_path / "config"), output_path=str(aggregator_results_path)
)

aggregator = af.Aggregator(directory=str(aggregator_results_path))

# Now, lets create a list of instances of the most-likely models of the final phase of each fit.
pipeline_name = "pipeline__lens_sie__source_inversion"
phase_name = "phase_3__source_inversion"

multi_nest_outputs = aggregator.filter(phase=phase_name).output

most_likely_model_instances = [
    out.most_probable_model_instance for out in multi_nest_outputs
]

# A model instance is simply the GalaxyModel of the pipeline that ran. That is, its just a list of galaxies, which we
# can pass to the objects we're used to using in PyAutoLens. Lets create the most-likely tracer of every fit and
# then plot its subplot.
most_likely_tracers = [
    al.Tracer.from_galaxies(galaxies=instance.galaxies)
    for instance in most_likely_model_instances
]

[aplt.tracer.subplot_tracer(tracer=tracer) for tracer in most_likely_tracers]

# Instances are just lists of galaxies, meaning that we can directly extract the quantities that are accessible in the
# Galaxy class. Lets print the Einstein mass of each of our most-likely lens galaxies.

# A model instance uses the model defined by a pipeline. The model is our list of galaxies and we can extract their
# parameters provided we know the galaxy names.
print("Most Likely Lens Einstein Masses:")
print(
    [
        instance.galaxies.lens.mass.einstein_mass
        for instance in most_likely_model_instances
    ]
)
print()

# Okay, lets now do something a bit more ambitious. Lets create a plot of the einstein_radius vs axis_ratio of each
# SIE mass profile, and lets include error bars at 3 sigma confidence.

most_probable_model_instances = [
    out.most_probable_model_instance for out in multi_nest_outputs
]
upper_error_instances = [
    out.model_errors_instance_at_upper_sigma_limit(sigma_limit=3.0)
    for out in multi_nest_outputs
]
lower_error_instances = [
    out.model_errors_instance_at_lower_sigma_limit(sigma_limit=3.0)
    for out in multi_nest_outputs
]

einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius
    for instance in most_probable_model_instances
]
einstein_radii_upper = [
    instance.galaxies.lens.mass.einstein_radius
    for instance in most_probable_model_instances
]
einstein_radii_lower = [
    instance.galaxies.lens.mass.einstein_radius
    for instance in most_probable_model_instances
]
axis_ratios = [
    instance.galaxies.lens.mass.axis_ratio for instance in most_probable_model_instances
]
axis_ratios_upper = [
    instance.galaxies.lens.mass.axis_ratio for instance in most_probable_model_instances
]
axis_ratios_lower = [
    instance.galaxies.lens.mass.axis_ratio for instance in most_probable_model_instances
]

print("Errors of SIE Einstein Radii:")
print([instance.galaxies.mass.einstein_radius for instance in upper_error_instances])
print([instance.galaxies.mass.einstein_radius for instance in lower_error_instances])
print()

# We can use the aggregator to create a list of the data-sets fitted by a pipeline. The results in this list will be in
# the same order as the non-linear outputs, meaning we can easily use their results to fit these data-sets.

pipeline_name = "pipeline_source__inversion__lens_sie_source_inversion"
phase_name = "phase_4__lens_sie__source_inversion"
phase_tag = "phase_tag__sub_2"

datasets = aggregator.filter(phase=phase_name, phase_tag=phase_tag).dataset
# datasets = aggregator.filter(pipeline=pipeline_name, name=phase_name, phase_tag=phase_tag).dataset

print("Datasets:")
print(datasets, "\n")

# We can plot instances of the dataset object:
# [aplt.imaging.subplot_imaging(imaging=dataset) for dataset in datasets]

# We can also load the masks used by an aggregator:
masks = aggregator.filter(phase=phase_name, phase_tag=phase_tag).mask
print("Masks:")
print(masks, "\n")

masks = [
    al.mask.circular(shape_2d=dataset.shape_2d, pixel_scales=0.1, radius=3.0)
    for dataset in datasets
]

# Next, we can use these masks to create the masked-imaging of every dataset, which we'll then use to perform a fit.

masked_imagings = [
    al.masked.imaging(dataset=dataset, mask=mask)
    for dataset, mask in zip(datasets, masks)
]

# Okay, so now lets use the most likely model (loaded using the code from the previous tutorial) to plot the best-fit
# model of every lens.
outputs = aggregator.filter(phase=phase_name, phase_tag=phase_tag).output
most_likely_model_instances = [out.most_probable_model_instance for out in outputs]

most_likely_fits = [
    al.fit(masked_dataset=masked_imaging, tracer=tracer)
    for masked_imaging, tracer in zip(masked_imagings, most_likely_tracers)
]
