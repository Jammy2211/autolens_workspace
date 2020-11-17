# %%
"""
__Aggregator 4: Grid Search__

"""

# %%
from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autolens as al

# %%
"""
Below, we set up the aggregator as we did in the previous tutorial.
"""

# %%
agg = af.Aggregator(directory=path.join("output", "aggregator", "grid_search"))

pipeline_name = "pipeline_subhalo__nfw"
name = "phase[1]__subhalo_search__source"

agg_grid_search = agg.filter(
    agg.phase == name,
    agg.pipeline == pipeline_name,
    agg.directory.contains("mass_sie__source_bulge__0"),
)

array = al.agg.grid_search_result_as_array(
    aggregator=agg_grid_search, use_log_evidences=True
)

# %%
"""
We are famaliar with filtering by pipeline name and phase name, so lets get the results of the `EllipticalPowerLaw` advanced 
pipeline.
"""

# %%
pipeline_name = "pipeline_mass__power_law"
name = "phase[1]__lens_power_law__source"

agg_power_law = agg.filter(agg.phase == name, agg.pipeline == pipeline_name)

print("Pipeline Name Filtered MultiNest Samples:")
print(list(agg_power_law.values("samples")))
print("Total Samples Objects = ", len(list(agg_power_law.values("samples"))), "\n")

# %%
"""
This gives 12 results, given that we fitted each of our 3 images 4 times using different pipeline settings.

Lets say we want only the fits that used the hyper galaxies functionality and included a shear. To get these results, 
we require a new filtering method based on the pipeline and phase tags of a given set of results. For this, we can 
filter based on the full path of a set of results, filtering for results that contain an input string. 

As usual, filtering creates a new aggregator.
"""

# %%

# This gives the 6 results with hyper galaxy fitting switch on.
agg_power_law_hyper_shear = agg_power_law.filter(
    agg_power_law.directory.contains("hyper_galaxies")
)

# This gives the 3 results from the 6 above that include a shear.
agg_power_law_hyper_shear = agg_power_law_hyper_shear.filter(
    agg_power_law_hyper_shear.directory.contains("with_shear")
)
