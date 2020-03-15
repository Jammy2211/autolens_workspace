# Once a pipeline has completed running, we have a set of results on our hard disk we manually inspect and analyse. If
# our dataset is large (e.g. images of many strong lenses) we will most likely run the same pipeline on the full sample,
# creating many folders on our hard disk. We'll quickly have too many results for it to be feasible to inspect and
# analsye results by sifting through the hard disk.

# PyAutoFit's aggregator tool allows us to load results in a Python script of Jupyter notebook. All we have to do is
# point the aggregator to the output directory from which we want to load results, which in this case will be the
# results of the first pipeline of this chapter.

### AUTOFIT + CONFIG SETUP ###

import autofit as af

# Setup the path to the autolens_workspace, using by filling in your path below.
workspace_path = "/path/to/user/autolens_workspace/"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace/"

# Setup the path to the config folder, using the autolens_workspace path.
config_path = workspace_path + "config"
output_path = workspace_path + "output"

# Use this path to explicitly set the config path and output path.
af.conf.instance = af.conf.Config(config_path=config_path)

# To set up the aggregator we simply pass it the folder of the results we want to load.
aggregator = af.Aggregator(directory=str(output_path))

# We can get the output of every non-linear search (e.g. all 3 phases) of every data-set fitted using the phase above.
non_linear_outputs = aggregator.output

# From here, we can inspect results as we please, for example printing the most likely model of every phase.

print([output.most_likely_instance for output in non_linear_outputs])

# And with that, we're done. So soon you ask? Surely the aggregator has a lot more that warrants discussion? You're
# right, however I don't believe it belongs in the HowToLens lectures. Only those who are looking to model large
# samples of lenses or fit many different models *need* to use the aggregator, and it is a tool that is better learnt
# through the day-to-day use of PyAutoLens.

# The full set of aggregator tutorials can be found at the location 'autolens_workspace/aggregator'. Here, you'll learn
# how to:

# - Use the aggregator to filter out results given a pipeline name or phase name.
# - Use the NonLinearOutput to produce many different results from the fit, including error estimates on parameters
#   and plots of the probability density function of parameters in 1D and 2D.
# - Reproduce visualizations of results, such as a tracer's images or the fit to a lens dataset.

# Even if you are only modeling a small sample of lenses, if you anticipate using PyAutoLens for the long-term I
# strongly recommend you begin using the aggregator to inspect and analyse your result. This is because it means you
# can perform all analyse in a Jupyter notebook, which as you already know is a flexible and versatile way to check
# results and make figures - its a better  more efficienct`workflow' in general.

# In HowToLelens, the main purpose of this tutorial was to make sure that you are aware of the aggregator's exsistance,
# and now you are!
