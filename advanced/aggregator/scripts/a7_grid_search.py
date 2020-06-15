# %%
"""
__Aggregator 4: Grid Search__

"""

# %%
from autoconf import conf
import autofit as af
import autolens as al
import autolens.plot as aplt

# %%
"""
Below, we set up the aggregator as we did in the previous tutorial.
"""

# %%
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"
output_path = f"{workspace_path}/output"
agg_results_path = f"{output_path}/aggregator/grid_search"

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=output_path
)

agg = af.Aggregator(directory=str(agg_results_path))

pipeline_name = "pipeline_subhalo__nfw"
phase_name = "phase_1__subhalo_search__source"

agg_grid_search = agg.filter(
    agg.phase == phase_name,
    agg.pipeline == pipeline_name,
    agg.directory.contains("lens_sie__source_sersic__0"),
)

array = al.agg.grid_search_result_as_array(
    aggregator=agg_grid_search, use_max_log_likelihoods=True
)

# %%
"""
We are famaliar with filtering by pipeline name and phase name, so lets get the results of the power-law advanced 
pipeline.
"""

# %%
pipeline_name = "pipeline_mass__power_law"
phase_name = "phase_1__lens_power_law__source"

agg_power_law = agg.filter(agg.phase == phase_name, agg.pipeline == pipeline_name)

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
