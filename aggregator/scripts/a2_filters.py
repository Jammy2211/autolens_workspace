# %%
"""
__Aggregator 2: Filters__

Lets suppose we had the results of other fits in the folder `output/aggregator`, and we *only* wanted fits which used
the phase defined in `phase_runner.py`. To avoid loading all the other results, we can use the aggregator`s filter
tool, which filters the results and provides us with only the results we want.

The filter provides us with the aggregator object we used in the previous tutorial, so can be used in an identical
fashion to tutorial 1.
"""


# %%
from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af

# %%
"""
First, set up the aggregator as we did in the previous tutorial.
"""

# %%
agg = af.Aggregator(directory="output/aggregator")

# %%
"""
We can first filter results to only include completed results. By including the `completed_only` input below, any 
results which are in the middle of a non-linear will be omitted and not loaded in the `Aggregator`.
"""

# %%
agg = af.Aggregator(directory="output/aggregator", completed_only=True)

# %%
"""
The simplest filter uses the name to filter all results in the output_path. Below, we use the phase name of the
phase used to fit our aggregator sample, providing us with all 3 results.
"""

# %%
name = "phase__aggregator"
agg_filter_phase = agg.filter(agg.phase == name)
samples_gen = agg_filter_phase.values("samples")

# %%
"""
As expected, this list retains 3 NestSamples objects.
"""

# %%
print("Phase Name Filtered NestedSampler Samples: \n\n")
print("Total Samples Objects = ", len(list(agg_filter_phase.values("samples"))), "\n\n")

# %%
"""
If we filtered using an incorrect phase name we would get no results:
"""

# %%
name = "phase__incorrect_name"
agg_filter_incorrect = agg.filter(agg.phase == name)
print("Incorrect Phase Name Filtered NestedSampler Samples: \n")
print(
    "Total Samples Objects = ",
    len(list(agg_filter_incorrect.values("samples"))),
    "\n\n",
)

# %%
"""
Alternatively, we can filter using strings, requiring that the string appears in the full path of the output
results. This is useful if you fit a samples of lenses where:

 - Multiple results, corresponding to different pipelines, phases and model-fits are stored in the same path.
 - Different runs using different `SettingsPhase` and `SetupPipeline` are in the same path.
 - Fits using different non-linear searches, with different settings, are contained in the same path.

The example below shows us using the contains filter to get the results of all 3 lenses. The contains method
only requires that the string is in the path structure, thus we do not need to specify the full phase name.
"""
agg_filter_contains = agg.filter(agg.directory.contains("phase__"))
print("Directory Contains Filtered NestedSampler Samples: \n")
print(
    "Total Samples Objects = ", len(list(agg_filter_contains.values("samples"))), "\n\n"
)

# %%
"""
If the model-fit was performed using a `Pipeline`, you can filter by the pipeline name to get results. 

The example output in this tutorial did not use a pipeline, so filtering by pipeline name removes all result.
"""
pipeline_name = "pipeline_name"
agg_filter_pipeline = agg.filter(agg.pipeline == pipeline_name)
print("Pipeline Name Filtered NestedSampler Samples: \n")
print(
    "Total Samples Objects = ", len(list(agg_filter_pipeline.values("samples"))), "\n\n"
)

# %%
"""
Finally, filters can be combined to load precisely only the result that you want, below we use all the above filters to 
load only the results of the fit to the first lens in our sample.
"""
name = "phase__aggregator"
agg_filter_multiple = agg.filter(
    agg.phase == name,
    agg.directory.contains("phase__"),
    agg.directory.contains("dynesty"),
    agg.directory.contains("mass_sie__source_bulge__0"),
)
print("Multiple Filter NestedSampler Samples: \n")
print()
print(
    "Total Samples Objects = ", len(list(agg_filter_multiple.values("samples"))), "\n\n"
)
