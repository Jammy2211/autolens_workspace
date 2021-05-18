"""
Database 4: Lens Models
=======================

This tutorial builds on the tutorial `a1_samples`, where we use the aggregator to load models from a non-linear
search and visualize and interpret results.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
from astropy import cosmology as cosmo

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autolens as al
import autolens.plot as aplt
import matplotlib.pyplot as plt

"""
First, set up the aggregator as we did in the previous tutorial.
"""
agg = af.Aggregator.from_database(path.join("output", "database.sqlite"))

"""
Next, lets create a list of instances of the maximum log likelihood models of each fit.
"""
ml_instances = [samps.max_log_likelihood_instance for samps in agg.values("samples")]

"""
A model instance contains a list of `Galaxy` instances, which is what we are using to passing to functions in 
PyAutoLens. Lets create the maximum log likelihood tracer of every fit.
"""
ml_tracers = [
    al.Tracer.from_galaxies(galaxies=instance.galaxies) for instance in ml_instances
]

print("Maximum Log Likelihood Tracers: \n")
print(ml_tracers, "\n")
print("Total Tracers = ", len(ml_tracers))

"""
Now lets plot their convergences, using a grid of 100 x 100 pixels (noting that this isn't` necessarily the grid used
to fit the data in the search itself).
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer in ml_tracers:
    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(convergence=True)

"""
Okay, so we can make a list of tracers and plot their convergences. However, we'll run into the same problem using 
lists which we discussed in the previous tutorial. If we had fitted hundreds of images we`d have hundreds of tracers, 
overloading the memory on our laptop.

We can again avoid using lists for any objects that could potentially be memory intensive, using generators.

The `fit` in the function below corresponds to a `Fit` object in the database, and it has access to many of the
objects we have been loading generators of via the `agg.values` method in previous tutorials.
"""


def make_tracer_generator(fit):

    samples = fit.value(name="samples")

    return al.Tracer.from_galaxies(
        galaxies=samples.max_log_likelihood_instance.galaxies
    )


"""
We `map` the function above using our aggregator to create a tracer generator.
"""
tracer_gen = agg.map(func=make_tracer_generator)

"""
We can now iterate over our tracer generator to make the plots we desire.
"""
grid = al.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.1)

for tracer in tracer_gen:

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(convergence=True, potential=True)

"""
Its cumbersome to always have to define a `make_tracer_generator` function to make a tracer generator, given that 
you`ll probably do the exact same thing in every Jupyter Notebook you ever write!

PyAutoLens`s aggregator module (imported as `agg`) has convenience methods to save you time and make your notebooks
cleaner.
"""
tracer_gen = al.agg.Tracer(aggregator=agg)

for tracer in tracer_gen:

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(convergence=True, potential=True)

"""
Because instances are just lists of galaxies we can directly extract attributes of the `Galaxy` class. Lets print 
the Einstein mass of each of our most-likely lens galaxies.

The model instance uses the model defined by a pipeline. In this pipeline, we called the lens galaxy `lens`.

For illustration, lets do this with a list first:
"""
print("Maximum Log Likelihood Lens Einstein Masses:")

for instance in ml_instances:

    einstein_mass = instance.galaxies.lens.einstein_mass_angular_from_grid(grid=grid)

    print("Einstein Mass (angular units) = ", einstein_mass)

    critical_surface_density = al.util.cosmology.critical_surface_density_between_redshifts_from(
        redshift_0=instance.galaxies.lens.redshift,
        redshift_1=instance.galaxies.source.redshift,
        cosmology=cosmo.Planck15,
    )

    einstein_mass_kpc = einstein_mass * critical_surface_density

    print("Einstein Mass (kpc) = ", einstein_mass_kpc)
    print("Einstein Mass (kpc) = ", "{:.4e}".format(einstein_mass_kpc))

    print(einstein_mass)

print()

"""
Now lets use a generator.

The `Fit` in the database includes the `instance` corresponding to the maximum log likelihood model. Below, you can see
we access this model and its `galaxies` to print the best fit model.
"""


def print_max_log_likelihood_mass(fit):

    einstein_mass = fit.instance.galaxies.lens.einstein_mass_angular_from_grid(
        grid=grid
    )

    print("Einstein Mass (angular units) = ", einstein_mass)

    critical_surface_density = al.util.cosmology.critical_surface_density_between_redshifts_from(
        redshift_0=fit.instance.galaxies.lens.redshift,
        redshift_1=fit.instance.galaxies.source.redshift,
        cosmology=cosmo.Planck15,
    )

    einstein_mass_kpc = einstein_mass * critical_surface_density

    print("Einstein Mass (kpc) = ", einstein_mass_kpc)
    print("Einstein Mass (kpc) = ", "{:.4e}".format(einstein_mass_kpc))


print()
print("Maximum Log Likelihood Lens Einstein Masses:")
agg.map(func=print_max_log_likelihood_mass)


"""
In the runner, we used the pickle_files input to search.run() to pass a .pickle file from the dataset folder to 
the database. 

Our strong lens dataset was created via a simulator script, so we passed the `Tracer` used to simulate the strong
lens, which was written as a .pickle file called `true_tracer.pickle` to the search to make it accessible in the 
database. This will allow us to directly compare the inferred lens model to the `truth`. 
"""
# true_tracers = [true_tracer for true_tracer in agg.values("true_tracer")]

print("Parameters used to simulate the lens dataset:")
# print(true_tracers)

"""
Finish.
"""
