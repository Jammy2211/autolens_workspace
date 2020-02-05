from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

# This tutorial builds on tutorial_1 of the aggregator autolens_workspace. Here, we will use the aggregator to loads
# models from a non-linear search and use them to visualize results and fits.

# Below, we set up the aggregator as we did in the previous tutorial.
workspace_path = Path(__file__).parent.parent
output_path = workspace_path / "output"
aggregator_results_path = output_path / "aggregator_sample"

af.conf.instance = af.conf.Config(
    config_path=str(workspace_path / "config"), output_path=str(aggregator_results_path)
)

aggregator = af.Aggregator(directory=str(aggregator_results_path))

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
most_likely_tracers = [
    al.Tracer.from_galaxies(galaxies=instance.galaxies)
    for instance in most_likely_model_instances
]
most_likely_fits = [
    al.fit(masked_dataset=masked_imaging, tracer=tracer)
    for masked_imaging, tracer in zip(masked_imagings, most_likely_tracers)
]
