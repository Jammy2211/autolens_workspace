# %%
"""
__Aggregator 4: Derived__

This tutorial describes how to estimate derived quantities from a model-fit, where a derived quantity is one which may
be used for the analysis and interpreation of results but is not explicitly a free parameter in the non-linear search.

An example is the total luminosity of the lens or source galaxy, or total mass of the lens galaxy. These quantities are
estimated by a PyAutoLens model-fit, but are estimated from a combination of lens model parameters.
"""

import autofit as af
import autolens as al
import autolens.plot as aplt

import numpy as np
import matplotlib.pyplot as plt

# %%
"""
Frist, we set up the aggregator as we did in the previous tutorial.
"""

# %%
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"
output_path = f"{workspace_path}/output"
agg_results_path = f"{output_path}/aggregator/beginner"

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=output_path
)

agg = af.Aggregator(directory=str(agg_results_path))

# %%
"""
Next, lets create a list of instances of the most-likely models of the final phase of each fit.
"""

# %%
pipeline_name = "pipeline__lens_sie__source_inversion"
phase_name = "phase_3__source_inversion"
agg_phase_3 = agg.filter(agg.phase == phase_name)

# %%
"""
To begin, lets compute the mass of a lens model, including the errors on the mass. In the previous tutorial, we saw that
the errors on a quantity like the mass model axis_ratio is simple, because it was sampled by the non-linear. Thus, to 
get errors on the axis ratio we can uses the Samples object to simply marginalize over all over parameters via the 
1D Probability Density Function (PDF).

But what if we want the errors on the Einstein Mass? This wasn't a free parameter in our model so we can't just 
marginalize over all other parameters.

Instead, we need to compute the Einstein mass of every lens model sampled by MultiNest and from this determine the 
PDF of the Einstein mass. When combining the different Einstein masses we weight each value by its MultiNest sampling 
probablity. This means that models which gave a poor fit to the data are downweighted appropriately.

Below, we get an instance of every MultiNest sample using the Samples, compute that models einstein mass, 
store them in a list and find the weighted median value with errors.

This function takes the list of Einstein mass values with their sample weights and computed the weighted mean and 
standard deviation of these values.
"""

# %%
def weighted_mean_and_standard_deviation(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- Numpy ndarrays with the same shape.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


# %%
"""
Now, we iterate over each Samples object, using every model instance to compute its mass. We combine these masses with 
the samples weights to the weighted mean Einstein Mass alongside its error.

Computing an Einstein mass takes a bit of time, so be warned this cell could run for a few minutes! To speed things 
up, you'll notice that we only perform the loop on samples whose probably is above 1.0e-4.
"""

# %%
def mass_error(agg_obj):

    samples = agg_obj.samples

    sample_masses = []

    for sample_index in range(samples.total_accepted_samples):

        if samples.weights[sample_index] > 1.0e-4:

            instance = samples.instance_from_sample_index(sample_index=sample_index)

            einstein_mass = instance.galaxies.lens.einstein_mass_in_units(
                redshift_object=instance.galaxies.lens.redshift,
                redshift_source=instance.galaxies.source.redshift,
            )

            sample_masses.append(einstein_mass)

    return weighted_mean_and_standard_deviation(
        values=sample_masses, weights=samples.weights
    )


einstein_masses, einstein_mass_errors = agg_phase_3.map(func=mass_error)

print("Einstein Masses:\n")
print(einstein_masses)
print("Einstein Mass Errors\n")
print(einstein_mass_errors)

# %%
"""
We can also iterate over every Fit of our results, to extracting derived information on the fit. Below, we reperform
every source reconstruction of the fit and ?
"""

# %%
fit_gen = al.agg.FitImaging(aggregator=agg_phase_3)

for fit in fit_gen:

    print(fit.inversion)
