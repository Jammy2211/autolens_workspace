# %%
"""
__Aggregator 4: Derived__

This tutorial describes how to estimate derived quantities from a model-fit, where a derived quantity is one which may
be used for the analysis and interpreation of results but is not explicitly a free parameter in the non-linear search.

An example is the total luminosity of the lens or source galaxy, or total mass of the lens galaxy. These quantities
are estimated by a PyAutoLens model-fit, but are estimated from a combination of lens model parameters.
"""

# %%
from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autolens as al
import numpy as np

# %%
"""
First, we set up the aggregator as we did in the previous tutorial.
"""

# %%
agg = af.Aggregator(directory="output/aggregator")

# %%
name = "phase__aggregator"
agg_phase3 = agg.filter(agg.phase == name)

# %%
"""
To begin, lets compute the axis ratio of a lens model, including the errors on the axis ratio. In the previous tutorials, 
we saw that the errors on a quantity like the elliptical_comps is simple, because it was sampled by the non-linear 
search. Thus, to get their we can uses the Samples object to simply marginalize over all over parameters via the 1D 
Probability Density Function (PDF).

But what if we want the errors on the axis-ratio? This wasn`t a free parameter in our model so we can`t just 
marginalize over all other parameters.

Instead, we need to compute the axis-ratio of every lens model sampled by the `NonLinearSearch` and from this determine 
the PDF of the axis-ratio. When combining the different axis-ratios we weight each value by its `weight`. For Dynesty,
the nested sampler we fitted our aggregator sample with, this downweights the model which gave lower likelihood fits.
For other `NonLinearSearch` methods (e.g. MCMC) the weights can take on a different meaning but can still be used for
combining different model results.

Below, we get an instance of every Dynesty sample using the `Samples`, compute that models axis-ratio, store them in a 
list and find the weighted median value with errors.

This function takes the list of axis-ratio values with their sample weights and computes the weighted mean and 
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
Now, we iterate over each Samples object, using every model instance to compute its axis-ratio. We combine these 
axis-ratios with the samples weights to give us the weighted mean axis-ratio and error.

To do this, we again use a generator. Whislt the axis-ratio is a fairly light-weight value, and this could be
performed using a list without crippling your comptuer`s memory, for other quantities this is not the case. Thus, for
computing derived quantities it is good practise to always use a generator.
"""

# %%
def axis_ratio_error_from_agg_obj(agg_obj):

    samples = agg_obj.samples

    axis_ratios = []

    for sample_index in range(samples.total_accepted_samples):

        instance = samples.instance_from_sample_index(sample_index=sample_index)

        axis_ratio = al.convert.axis_ratio_from(
            elliptical_comps=instance.galaxies.mass.elliptical_comps
        )

        axis_ratios.append(axis_ratio)

    return weighted_mean_and_standard_deviation(
        values=axis_ratios, weights=samples.weights
    )


axis_ratio, axis_ratio_error = agg_phase3.map(func=axis_ratio_error_from_agg_obj)

print("Axis Ratio:\n")
print(axis_ratio)
print("Axis Ratio Error\n")
print(axis_ratio_error())
stop


def axis_ratio_error(agg_obj):

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


# %%
"""
We can also iterate over every Fit of our results, to extracting derived information on the fit. Below, we reperform
every source reconstruction of the fit and ?
"""

# %%
fit_gen = al.agg.FitImaging(aggregator=agg_phase[3])

for fit in fit_gen:

    print(fit.inversion)
