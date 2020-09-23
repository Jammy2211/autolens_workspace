# %%
"""
__Aggregator 3: Lens Models__

This tutorial builds on the tutorial `a1_samples`, where we use the aggregator to load models from a non-linear
search and visualize and interpret results.
"""

# %%
from autoconf import conf
import autofit as af
import autolens as al
import autolens.plot as aplt

import matplotlib.pyplot as plt

# %%
"""
First, set up the aggregator as we did in the previous tutorial.
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
Next, lets create a list of instances of the maximum log likelihood models of each fit. Although we don`t need to use
the aggregator`s filter tool, we'll use it below (and in all tutorials here after) so you are used to seeing it, as
for general PyAutoLens use it will be important to filter your results!
"""

# %%
phase_name = "phase__aggregator"
agg_filter = agg.filter(agg.phase == phase_name)

ml_instances = [
    samps.max_log_likelihood_instance for samps in agg_filter.values("samples")
]

# %%
"""
A model instance contains a list of `Galaxy` instances, which is what we are using to passing to functions in 
PyAutoLens. Lets create the maximum log likelihood tracer of every fit.
"""

# %%
ml_tracers = [
    al.Tracer.from_galaxies(galaxies=instance.galaxies) for instance in ml_instances
]

print("Maximum Log Likelihood Tracers: \n")
print(ml_tracers, "\n")
print("Total Tracers = ", len(ml_tracers))

# %%
"""
Now lets plot their convergences, using a grid of 100 x 100 pixels (noting that this isn`t` necessarily the grid used
to fit the data in the phase itself).
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.1)

for tracer in ml_tracers:
    aplt.Tracer.convergence(tracer=tracer, grid=grid)

# %%
"""
Okay, so we can make a list of tracers and plot their convergences. However, we'll run into the same problem using 
lists which we discussed in the previous tutorial. If we had fitted hundreds of images we`d have hundreds of tracers, 
overloading the memory on our laptop.

We can again avoid using lists for any objects that could potentially be memory intensive, using generators.
"""

# %%
def make_tracer_generator(agg_obj):

    output = agg_obj.samples

    # This uses the output of one instance to generate the tracer.
    return al.Tracer.from_galaxies(galaxies=output.max_log_likelihood_instance.galaxies)


# %%
"""
# We "map" the function above using our aggregator to create a tracer generator.
"""

# %%
tracer_gen = agg_filter.map(func=make_tracer_generator)

# %%
"""
We can now iterate over our tracer generator to make the plots we desire.
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.1)

for tracer in tracer_gen:

    aplt.Tracer.convergence(tracer=tracer, grid=grid)
    aplt.Tracer.potential(tracer=tracer, grid=grid)

# %%
"""
Its cumbersome to always have to define a `make_tracer_generator` function to make a tracer generator, given that 
you`ll probably do the exact same thing in every Jupyter Notebook you ever write!

PyAutoLens`s aggregator module (imported as `agg`) has convenience methods to save you time and make your notebooks
cleaner.
"""

# %%
tracer_gen = al.agg.Tracer(aggregator=agg_filter)

for tracer in tracer_gen:
    aplt.Tracer.convergence(tracer=tracer, grid=grid)
    aplt.Tracer.potential(tracer=tracer, grid=grid)

# %%
"""
Because instances are just lists of galaxies we can directly extract attributes of the `Galaxy` class. Lets print 
the Einstein mass of each of our most-likely lens galaxies.

The model instance uses the model defined by a pipeline. In this pipeline, we called the lens galaxy `lens`.

For illustration, lets do this with a list first:
"""

# %%
print("Maximum Log Likelihood Lens Einstein Masses:")
for instance in ml_instances:
    einstein_mass = instance.galaxies.lens.einstein_mass_in_units(
        redshift_object=instance.galaxies.lens.redshift,
        setup.redshift_source=instance.galaxies.source.redshift,
    )
    print(einstein_mass)
print()

# %%
"""
Now lets use a generator.
"""

# %%
def print_max_log_likelihood_mass(agg_obj):

    output = agg_obj.samples

    einstein_mass = output.instance.galaxies.lens.einstein_mass_in_units(
        redshift_object=output.instance.galaxies.lens.redshift,
        setup.redshift_source=output.instance.galaxies.source.redshift,
    )
    print(einstein_mass)


print("Maximum Log Likelihood Lens Einstein Masses:")
agg_filter.map(func=print_max_log_likelihood_mass)

# %%
"""
Lets next do something a bit more ambitious. Lets create a plot of the einstein_radius vs axis_ratio of each 
_EllipticalIsothermal_ `MassProfile`.

These plots don`t use anything too memory intensive (like a tracer) so we are fine to go back to lists for this.
"""

# %%
mp_instances = [samps.median_pdf_instance for samps in agg_filter.values("samples")]
mp_einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius for instance in mp_instances
]
mp_elliptical_comps = [
    instance.galaxies.lens.mass.elliptical_comps for instance in mp_instances
]

mp_axis_ratios = [
    al.convert.axis_ratio_from(elliptical_comps=ell) for ell in mp_elliptical_comps
]

print(mp_einstein_radii)
print(mp_axis_ratios)

plt.scatter(mp_einstein_radii, mp_axis_ratios, marker="x")
plt.show()

# %%
"""
Now lets also include error bars at 3 sigma confidence.
"""

# %%
ue3_instances = [
    samps.error_instance_at_upper_sigma(sigma=3.0)
    for samps in agg_filter.values("samples")
]
le3_instances = [
    samps.error_instance_at_lower_sigma(sigma=3.0)
    for samps in agg_filter.values("samples")
]

ue3_einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius for instance in ue3_instances
]
le3_einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius for instance in le3_instances
]
ue3_elliptical_comps = [
    instance.galaxies.lens.mass.elliptical_comps for instance in ue3_instances
]
le3_elliptical_comps = [
    instance.galaxies.lens.mass.elliptical_comps for instance in le3_instances
]

ue3_axis_ratios = [
    al.convert.axis_ratio_from(elliptical_comps=ell) for ell in ue3_elliptical_comps
]
le3_axis_ratios = [
    al.convert.axis_ratio_from(elliptical_comps=ell) for ell in le3_elliptical_comps
]

plt.errorbar(
    x=mp_einstein_radii,
    y=mp_axis_ratios,
    marker=".",
    linestyle="",
    xerr=[le3_einstein_radii, ue3_einstein_radii],
    yerr=[le3_axis_ratios, ue3_axis_ratios],
)
plt.show()

# %%
"""
In the phase_runner, we used the pickle_files input to phase.run() to pass a .pickle file from the dataset folder to 
the `Aggregator` pickles folder. 

Our strong lens dataset was created via a simulator script, so we passed the `Tracer` used to simulate the strong
lens, which was written as a .pickle file called `true_tracer.pickle` to the phase to make it accessible in the 
_Aggregator_. This will allow us to directly compare the inferred lens model to the `truth`. 

You should checkout `autolens_workspace/advanced/aggregator/phase_runner.py` to see how this was performed.
"""

true_tracers = [true_tracer for true_tracer in agg_filter.values("true_tracer")]

print("Parameters used to simulate first Aggregator dataset:")
print(true_tracers[0])
