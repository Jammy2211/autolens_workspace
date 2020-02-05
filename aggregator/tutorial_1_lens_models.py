from pathlib import Path

import autofit as af
import autoarray.plot as aplt

### INTRODUCTION ####

# After fitting a large suite of data with the same pipeline, the aggregator allows us to load the results and
# manipulate / plot them using a Python script or Jupyter notebook.

# In '/autolens_workspace/aggregator/runner.py' we fitted 3 strong lenses which were simulated using different lens
# models. We fitted each image with the pipelines:

# 'autolens_workspace/pipelines/advanced/no_lens_light/source/parametric/lens_sie__source_sersic.py'
# 'autolens_workspace/pipelines/advanced/no_lens_light/source/inversion/from_parametric/lens_sie__source_inversion.py'

### FILE OUTPUT ####

# The results of this fit are in the '/output/aggregator_sample' folder. First, take a look in this folder. Provided
# you haven't rerun the runner, you'll notice that all the results (e.g. optimizer, optimizer_backup, model.results,
# images, etc.) are in .zip files, as opposed to folders that can be instantly accessed.

# This is because when the lens model pipelines were run, the 'remove_files' option in the 'config/general.ini' was set
# to  True, such that all results (other than the .zip file) were removed. This feature is implemented because
# super-computers often have a limit on the number of files allowed per.

# Bare in mind the fact that all results are in .zip files - we'll come back to this point in a second.

### AGGREGATOR ###

# We can load the results of all 3 pipeline runs using the aggregator, which will then allow us to manipulate those
# results in this Python script or a Jupyter notebook to plot figures, interpret results, check specific values, etc.

# To begin, we setup the path to the output path we want to load results from, which in this case is the folder
# 'autolens_workspace/output/aggregator_sample'.
workspace_path = Path(__file__).parent.parent
output_path = workspace_path / "output"
aggregator_results_path = output_path + "/aggregator_sample"

# Now we'll use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(
    config_path=str(workspace_path / "config"), output_path=str(aggregator_results_path)
)

# To set up the aggregator we simply pass it the folder of the results we want to load.
aggregator = af.Aggregator(directory=str(aggregator_results_path))

# Before we continue, take another look at the output folder. The .zip files containing results have now all been
# unzipped, such that the resultls are readily accessible on your laptop for navigation. This means you can run fits to
# many lenses on a super computer and easily unzip all the results on your computer afterwards via the aggregator.

### MODEL RESULTS ###

# We can now create a list of the 'non-linear outpus' of every fit, where an instance of the Output class acts as an
# interface between the results of the non-linear fit on your hard-disk and Python.

# The fits to each lens used MultiNest, so we will create a list of instances of the MultiNestOutput class (if a
# different non-linear sampler were used the appropriate non-linear output class would be used).
multi_nest_outputs = aggregator.output

# When we print this list of outputs, we will see 15 different MultiNestOutput instances. These corresponded to all 5
# phases of all 3 fits.
print("MultiNest Outputs:")
print(multi_nest_outputs, "\n")

# Lets get rid of the results of the initialization pipeline by passing the name of the main pipeline we want to
# load the results of to the aggregator's filter method.
pipeline_name = "pipeline_source__inversion__lens_sie_source_inversion"
multi_nest_outputs = aggregator.filter(pipeline=pipeline_name).output

# As expected, this list now has only 12 MultiNestOutputs.
print("Pipeline Name Filtered MultiNest Outputs:")
print(multi_nest_outputs, "\n")

# Thats still a lot of outputs though! We can filter by phase name to get just the results of fitting the final phase
# in our pipelines.
phase_name = "phase_4__lens_sie__source_inversion"
multi_nest_outputs = aggregator.filter(
    pipeline=pipeline_name, phase_name=phase_name
).output

# As expected, this list now has only 3 MultiNestOutputs.
print("Phase Name Filtered MultiNest Outputs:")
print(multi_nest_outputs, "\n")

# We can, use these outputs to create a list of the most-likely (e.g. highest likelihood) model of each fit to our
# three images (in this phase).
most_likely_model_parameters = [
    out.most_probable_model_parameters for out in multi_nest_outputs
]
print("Most Likely Model Parameter Lists:")
print(most_likely_model_parameters, "\n")

# This provides us with lists of all model parameters. However, this isn't that much use - which values correspond
# to which parameters?

# Its more useful to create the model instance of every fit.
most_likely_model_instances = [
    out.most_probable_model_instance for out in multi_nest_outputs
]
print("Most Likely Model Instances:")
print(most_likely_model_instances, "\n")

# A model instance uses the model defined by a pipeline. The model is our list of galaxies and we can extract their
# parameters provided we know the galaxy names.
print("Most Likely SIE Einstein Radii:")
print(
    [instance.galaxies.mass.einstein_radius for instance in most_likely_model_instances]
)
print()

# We can also access the 'most probable' model, which is the model computed by marginalizing over the MultiNest samples
# of every parameter in 1D and taking the median of this PDF.
most_probable_model_parameters = [
    out.most_probable_model_parameters for out in multi_nest_outputs
]
most_probable_model_instances = [
    out.most_probable_model_instance for out in multi_nest_outputs
]

print("Most Probable Model Parameter Lists:")
print(most_probable_model_parameters, "\n")
print("Most probable Model Instances:")
print(most_probable_model_instances, "\n")
print("Most Probable SIE Einstein Radii:")
print(
    [
        instance.galaxies.mass.einstein_radius
        for instance in most_probable_model_instances
    ]
)
print()

# We can compute the upper and lower model errors at a given sigma limit.
upper_errors = [
    out.model_errors_at_upper_sigma_limit(sigma_limit=3.0) for out in multi_nest_outputs
]
upper_error_instances = [
    out.model_errors_instance_at_upper_sigma_limit(sigma_limit=3.0)
    for out in multi_nest_outputs
]
lower_errors = [
    out.model_errors_at_lower_sigma_limit(sigma_limit=3.0) for out in multi_nest_outputs
]
lower_error_instances = [
    out.model_errors_instance_at_lower_sigma_limit(sigma_limit=3.0)
    for out in multi_nest_outputs
]

print("Errors Lists:")
print(upper_errors, "\n")
print(lower_errors, "\n")
print("Errors Instances:")
print(upper_error_instances, "\n")
print(lower_error_instances, "\n")
print("Errors of SIE Einstein Radii:")
print([instance.galaxies.mass.einstein_radius for instance in upper_error_instances])
print([instance.galaxies.mass.einstein_radius for instance in lower_error_instances])
print()

# We can load the "model_results" of all phases, which is string that summarizes every fit's lens model providing
# quick inspection of all results.
results = aggregator.filter(pipeline=pipeline_name).model_results
print("Model Results Summary:")
print(results, "\n")
