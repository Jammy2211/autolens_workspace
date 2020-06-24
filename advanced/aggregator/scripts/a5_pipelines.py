# %%
"""
__Aggregator 4: Advanced__

The previous tutorials used beginner pipelines, whose results are distributed in one folder. Advanced pipelines results
are output in a more complex path structure with multiple pipelines. Furthemore, advanced pipelines allow us to fit
many different lens model parameterizations with the results output in strucuted paths depending on the phase and
pipeline tags. This tutorial explains how to use the aggregator for such complex outputs.

In '/autolens_workspace/aggregator/setup/advanced_runner.py' we fit our 3 images with the following 3 pipelines:

'autolens_workspace/pipelines/advanced/no_lens_light/source/parametric/lens_sie__source_sersic.py'
'autolens_workspace/pipelines/advanced/no_lens_light/source/inversion/from_parametric/lens_sie__source_inversion.py'
'autolens_workspace/pipelines/advanced/no_lens_light/mass/power_law/lens_power_law__source_inversion.py'

Each set of 3 images is fitted 4 separate times, with the following variants:

- With the General setup hyper_galaxies=False and with no_shear=False.
- With the General setup hyper_galaxies=True and with no_shear=False.
- With the General setup hyper_galaxies=False and with no_shear=True.
- With the General setup hyper_galaxies=True and with no_shear=True.

The results of these fits are in the '/output/aggregator_sample_advanced' folder. Pipeline tagging has lead to many
different results in a complex path structure that depends on the setup of the pipeline.
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
agg_results_path = f"{output_path}/aggregator/advanced"

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=output_path
)

agg = af.Aggregator(directory=str(agg_results_path))

# %%
"""
In this example, we only fitted the 3 images using one pipeline. But suppose we used multiple pipelines, like we do in 
the advanced pipelines. In this case, the aggregator would load the NestSampless of all fits of all phases of all 
pipelines!

In such circumstances, we can filter by phase name and pipeline name.
"""

# %%
pipeline_name = "pipeline__lens_sie__source_inversion"
agg_filter = agg.filter(agg.phase == phase_name, agg.pipeline == pipeline_name)
samples_gen = agg_filter.values("samples")

# %%
"""
As expected, this list again has 3 NestSampless.
"""

# %%
print("Pipeline Name Filtered MultiNest Samples: \n")
print(samples_gen)
print()
print("Ttotal Samples Objects = ", len(list(agg_filter.values("samples"))), "\n")

# %%
pipeline_name = "pipeline_mass__power_law"
phase_name = "phase_1__lens_power_law__source"

agg_power_law = agg.filter(agg.phase == phase_name, agg.pipeline == pipeline_name)

print("Pipeline Name Filtered MultiNest Samples:")
print(list(agg_power_law.values("samples")))
print("Ttotal Samples Objects = ", len(list(agg_power_law.values("samples"))), "\n")

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

# %%
"""
This aggregator can now be used, as usual, to make plots of quantities like the fit.
"""

# %%
fit_gen = al.agg.FitImaging(aggregator=agg_power_law_hyper_shear)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)


# %%
"""          
When there are many results in one directory, the best way to handle this is to create multiple aggregators using the
filter method above. Below, we create aggregators containing the results of fits to all 3 images which we then plot 
so we can compare the different fits.

Runs without hyper-galaxies or shear turned on do not tag the results with text. To get these results via path 
filtering we have to input the whole pipeline or phase tag, including the '/' to mark the end of the tag in the path 
string.
"""

# %%
agg_tmp = agg_power_law.filter(agg.directory.contains("general/"))
agg_power_law_no_shear = agg_tmp.filter(agg_tmp.directory.contains("with_shear"))

agg_tmp = agg_power_law.filter(agg.directory.contains("general/"))
agg_power_law_shear = agg_tmp.filter(agg_power_law.directory.contains("with_shear"))

agg_tmp = agg_power_law.filter(agg_power_law.directory.contains("hyper_galaxies"))
agg_power_law_hyper_no_shear = agg_tmp.filter(
    agg_tmp.directory.contains("source__pix_voro_mag__reg_const/")
)

agg_tmp = agg_power_law.filter(agg_power_law.directory.contains("hyper_galaxies"))
agg_power_law_hyper_shear = agg_tmp.filter(
    agg_tmp.directory.contains("source__pix_voro_mag__reg_const/")
)

fit_gen = al.agg.FitImaging(aggregator=agg_tmp)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

fit_gen = al.agg.FitImaging(aggregator=agg_power_law_shear)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

fit_gen = al.agg.FitImaging(aggregator=agg_power_law_hyper_no_shear)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

fit_gen = al.agg.FitImaging(agg_power_law_hyper_shear)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)

# %%
"""          
What if we want an aggregator which instead contains the 4 variant fits to 1 image? 

We can apply directory filtering using image names to achieve this.
"""

# %%
agg_dataset_0 = agg_power_law.filter(
    agg_power_law.directory.contains("lens_sie__source_sersic__0")
)

fit_gen = al.agg.FitImaging(aggregator=agg_dataset_0)

for fit in fit_gen:
    aplt.FitImaging.subplot_fit_imaging(fit=fit)
