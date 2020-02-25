import os

import autofit as af
import autolens as al
import autoarray.plot as aplt

# The previous tutorials used a beginner pipeline's results, which were all distributed in one folder for the pipeline.
# If you're used to using advanced pipelines and runners you'll know the path structure of the output gets a bit
# more complicated.
#
# If we only run one advanced pipeline, this isn't too tricky to deal with. However, in
# '/autolens_workspace/aggregator/setup/advanced_runner.py' we fitted our 3 images not only with these 3 pipeline:

# 'autolens_workspace/pipelines/advanced/no_lens_light/source/parametric/lens_sie__source_sersic.py'
# 'autolens_workspace/pipelines/advanced/no_lens_light/source/inversion/from_parametric/lens_sie__source_inversion.py'
# 'autolens_workspace/pipelines/advanced/no_lens_light/mass/power_law/lens_power_law__source_inversion.py'

# But 4 separate times, with the following variants:

# - With the GeneralSetup hyper_galaxies=False and with no_shear=False.
# - With the GeneralSetup hyper_galaxies=True and with no_shear=False.
# - With the GeneralSetup hyper_galaxies=False and with no_shear=True.
# - With the GeneralSetup hyper_galaxies=True and with no_shear=True.

# The results of these fits are in the '/output/aggregator_sample_advanced' folder. As you can see, the pipeline
# tagging has lead to a lot of different results depending on the setup of the pipeline. In this tutorial, we'll
# learn how to use the aggregator to filter through all these different variants!

# We begin as normal, pointing the aggregator to the output path of out results
# 'autolens_workspace/output/aggregator_sample_advanced'.
workspace_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = workspace_path + "output"
aggregator_results_path = output_path + "/aggregator_sample_advanced"

af.conf.instance = af.conf.Config(
    config_path=str(workspace_path + "/config"), output_path=str(output_path)
)

aggregator = af.Aggregator(directory=str(aggregator_results_path))

# Okay, so we are used to filtering by pipeline name and phase name. This is more than sufficient to get us results of
# an advanced pipeline.

pipeline_name = "pipeline_mass__power_law"
phase_name = "phase_1__lens_power_law__source"
outputs = aggregator.filter(pipeline=pipeline_name).output

# As expected, this list has 3 MultiNestOutputs.
print("Pipeline Name Filtered MultiNest Outputs:")
print(outputs)
print("Total Outputs = ", len(outputs), "\n")
