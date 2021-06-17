"""
Database 3: Queries
===================

Suppose we have the results of many fits in the `output` folder and we only wanted to load and inspect a specific set
of model-fits (e.g. the results of `tutorial_1_introduction`). We can use the database's querying tools to only load
the results we are interested in.

The database also supports advanced querying, so that specific model-fits (e.g., which fit a certain model or dataset)
can be loaded.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al


"""
__Database File__

First, we set up the aggregator like we did in the previous tutorial. However, we can also filter results to only 
include completed results. By including the `completed_only` input below, any results which are in the middle of a 
non-linear will be omitted and not loaded in the `Aggregator`.

For these tutorials, we only performed 3 model-fits which ran to completion, so this does not remove any results. For
general database use when you may have many model-fits running simultaneously, this filter can prove useful.
"""
agg = af.Aggregator.from_database(
    path.join("output", "database.sqlite"), completed_only=True
)

"""
__Unique Tag__

We can use the `Aggregator`'s to query the database and return only specific fits that we are interested in. We first 
do this, using the `unique_tag` which we can query to load the results of a specific `dataset_name` string we 
input into the model-fit's search. 

By querying using the string `mass_sie__source_sersic__1` the model-fit to only the second strong lens is returned:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "mass_sie__source_sersic__1")
samples_gen = agg_query.values("samples")

"""
As expected, this list now has only 1 DynestySamples corresponding to the second dataset.
"""
print("Directory Filtered DynestySampler Samples: \n")
print("Total Samples Objects via unique tag = ", len(samples_gen), "\n\n")

"""
If we query using an incorrect dataset name we get no results:
"""
unique_tag = agg.search.unique_tag
agg_query = agg.query(unique_tag == "incorrect_name")
samples_gen = agg_query.values("samples")

"""
__Search Name__

We can also use the `name` of the search used to fit to the model as a query. 

In this example, all three fits used the same search, which had the `name` `database_example`. Thus, using it as a 
query in this example is somewhat pointless. However, querying based on the search name is very useful for model-fits
which use search chaining (see chapter 3 **HowToLens**), where the results of a particular fit in the chain can be
instantly loaded.

As expected, this query contains all 3 results.
"""
name = agg.search.name
agg_query = agg.query(name == "database_example")
print("Total Queried Results via search name = ", len(agg_query), "\n\n")

"""
__Model Queries__

We can also query based on the model fitted. 

For example, we can load all results which fitted an `EllIsothermal` model-component, which in this simple 
example is all 3 model-fits.

The ability to query via the model is extremely powerful. It enables a user to fit many lens models to large samples 
of lenses efficiently load and inspect the results. 

[Note: the code `agg.galaxies.lens.mass` corresponds to the fact that in the `Model` we named the model components 
`galaxies`, `lens` and `mass`. If the `Model` had used a different name the code below would change correspondingly. 
Models with multiple galaxies are therefore easily accessed via the database.]
"""
lens = agg.galaxies.lens
agg_query = agg.query(lens.mass == al.mp.EllIsothermal)
print("Total Samples Objects via `EllIsothermal` model query = ", len(agg_query), "\n")

"""
Queries using the results of model-fitting are also supported. Below, we query the database to find all fits where the 
inferred value of `sersic_index` for the `EllSersic` of the source's bulge is less than 3.0 (which returns only 
the first of the three model-fits).
"""
bulge = agg.galaxies.source.bulge
agg_query = agg.query(bulge.sersic_index < 3.0)
print(
    "Total Samples Objects In Query `source.bulge.sersic_index < 3.0` = ",
    len(agg_query),
    "\n",
)

"""
__Logic__

Advanced queries can be constructed using logic, for example we below we combine the two queries above to find all
results which fitted an `EllIsothermal` mass model AND (using the & symbol) inferred a value of sersic index of 
less than 3.0 for the source's bulge. 

The OR logical clause is also supported via the symbol |.
"""
mass = agg.galaxies.lens.mass
agg_query = agg.query(
    (mass == al.mp.EllIsothermal) & (mass.einstein_radius_elliptical > 1.0)
)
print(
    "Total Samples Objects In Query `EllIsothermal and einstein_radius > 3.0` = ",
    len(agg_query),
    "\n",
)

"""
Finished.
"""
