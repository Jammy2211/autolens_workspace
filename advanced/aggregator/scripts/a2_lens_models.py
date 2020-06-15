# %%
"""
__Aggregator 2: Lens Models__

This tutorial builds on tutorial_1 of the aggregator autolens_workspace. Here, we use the aggregator to load models
from a non-linear search and visualize and interpret results.
"""

# %%
from autoconf import conf
import autofit as af
import autolens as al
import autolens.plot as aplt

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

ml_instances = [
    samps.max_log_likelihood_instance for samps in agg_phase_3.values("samples")
]

# %%
"""
A model instance is a _Galaxy_ instance of the pipeline's _GalaxyModel_'s. So, its just a list of galaxies which we can 
pass to functions in PyAutoLens. Lets create the most-likely tracer of every fit...
"""

# %%
ml_tracers = [
    al.Tracer.from_galaxies(galaxies=instance.galaxies) for instance in ml_instances
]

print("Most Likely Tracers: \n")
print(ml_tracers, "\n")
print("Total Tracers = ", len(ml_tracers))

# %%
"""
... and then plot their convergences.

We'll just use a grid of 100 x 100 pixels for now, and cover later how we use the actual grid of the data.
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.1)

for tracer in ml_tracers:
    aplt.Tracer.convergence(tracer=tracer, grid=grid)

# %%
"""
Okay, so we can make a list of tracers and plot their convergences. However, we'll run into the same problem using 
lists which we discussed in the previous tutorial. If we had fitted hundreds of images we'd have hundreds of tracers, 
overloading the memory on our laptop.

We will again avoid using lists for any objects that could potentially be memory intensive, using generators once 
again.
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
tracer_gen = agg_phase_3.map(func=make_tracer_generator)

# %%
"""
We can now iterate over our tracer generator to make the plots we desire. 

(We'll explain how to load the grid via the aggregator in the next tutorial)
"""

# %%
grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.1)

for tracer in tracer_gen:

    aplt.Tracer.convergence(tracer=tracer, grid=grid)
    aplt.Tracer.potential(tracer=tracer, grid=grid)

# %%
"""
Its cumbersome always have to define a 'make_tracer_generator' function to make a tracer generator - give that you'll
probably do the exact same thing in every Jupyter Notebook you ever write!

PyAutoLens's aggregator module (accessed as 'agg') has a convenience method to save you time and make your notebooks
cleaner.
"""

# %%
tracer_gen = al.agg.Tracer(aggregator=agg_phase_3)

for tracer in tracer_gen:
    aplt.Tracer.convergence(tracer=tracer, grid=grid)
    aplt.Tracer.potential(tracer=tracer, grid=grid)

# %%
"""
Because instances are just lists of galaxies we can directly extract attributes of the _Galaxy_ class. Lets print 
the Einstein mass of each of our most-likely lens galaxies.

The model instance uses the model defined by a pipeline. In this pipeline, we called the lens galaxy 'lens'.

For illustration, lets do this with a list first:
"""

# %%
print("Most Likely Lens Einstein Masses:")
for instance in ml_instances:
    einstein_mass = instance.galaxies.lens.einstein_mass_in_units(
        redshift_object=instance.galaxies.lens.redshift,
        redshift_source=instance.galaxies.source.redshift,
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
        redshift_source=output.instance.galaxies.source.redshift,
    )
    print(einstein_mass)


print("Most Likely Lens Einstein Masses:")
agg_phase_3.map(func=print_max_log_likelihood_mass)

# %%
"""
Lets next do something a bit more ambitious. Lets create a plot of the einstein_radius vs axis_ratio of each SIE mass 
profile.

These plots don't use anything too memory intensive - like a tracer - so we are fine to go back to lists for this.
"""

# %%
mp_instances = [samps.median_pdf_instance for samps in agg_phase_3.values("samples")]
mp_einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius for instance in mp_instances
]
mp_axis_ratios = [instance.galaxies.lens.mass.axis_ratio for instance in mp_instances]

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
    for samps in agg_phase_3.values("samples")
]
le3_instances = [
    samps.error_instance_at_lower_sigma(sigma=3.0)
    for samps in agg_phase_3.values("samples")
]

ue3_einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius for instance in ue3_instances
]
le3_einstein_radii = [
    instance.galaxies.lens.mass.einstein_radius for instance in le3_instances
]
ue3_axis_ratios = [instance.galaxies.lens.mass.axis_ratio for instance in ue3_instances]
le3_axis_ratios = [instance.galaxies.lens.mass.axis_ratio for instance in le3_instances]

plt.errorbar(
    x=mp_einstein_radii,
    y=mp_axis_ratios,
    marker=".",
    linestyle="",
    xerr=[le3_einstein_radii, ue3_einstein_radii],
    yerr=[le3_axis_ratios, ue3_axis_ratios],
)
plt.show()
