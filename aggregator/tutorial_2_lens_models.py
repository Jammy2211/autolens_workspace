from pathlib import Path

import autofit as af
import autolens as al
import autolens.plot as aplt

import matplotlib.pyplot as plt

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
    instance.galaxies.lens.mass.einstein_radius for instance in upper_error_instances
]
einstein_radii_lower = [
    instance.galaxies.lens.mass.einstein_radius for instance in lower_error_instances
]
axis_ratios = [
    instance.galaxies.lens.mass.axis_ratio for instance in most_probable_model_instances
]
axis_ratios_upper = [
    instance.galaxies.lens.mass.axis_ratio for instance in upper_error_instances
]
axis_ratios_lower = [
    instance.galaxies.lens.mass.axis_ratio for instance in lower_error_instances
]

plt.errorbar(
    x=einstein_radii, y=axis_ratios, xerr=einstein_radii_upper, yerr=axis_ratios_upper
)

# The final thing we might want to do is compute the errors on a value that wasn't a free parameter in our model fit.
# For example, getting the errors on the axis_ratios above is simple. MultiNest knows the samplers of the lens model,
# including the axis ratios, and it simply marginalizes over all over parameters to produce the PDF of the
# axis ratios.

# But what if we want the errors on the Einstein Mass? This wasn't a free parameter in our model and we are thus
# not able to simply marginalize over all other parameters.

# Instead, we need to compute the Einstein mass of every lens model sampled by MultiNest and from this determine the
# PDF of the Einstein mass. When combining the different Einstein mass we weight each value by the sampling probablity
# associated to that model by MultiNest, such that models that gave a poor fit to the data are downweighted appropriately.
