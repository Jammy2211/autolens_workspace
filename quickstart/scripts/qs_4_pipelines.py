### PIPELINES ###

# We just performed strong lens modeling using a 'Phase', which inferred the lens and source parameters using the
# non-linear seach MultiNest. A pipeline is a series of phases that are linked together, performing a chain of
# non-linear searches. This allows us to break-down fitting of a lens mode. So, why break the analysis down into
# multiple phases instead of using just 1 phase?

# For complex lens models fitting all lens model parameters without any initialization leads to either slow and
# inefficient lens modeling or infers an incorrect lens model. Pipelines circumvent these problems by initially fitting
# simple lens models (e.g. with fewer parameters) and gradually making the model that is fitted more complex. Crucially,
# a pipeline uses the results of the earlier phases to tell the non-linear search where to look in later phases.

# Open the script 'autolens_workspace/pipelines/simple/lens_sersic_sie__source_sersic.py'

# First, read the pipeline, which uses phases to model a lens. The introduces a number of new concepts, such as
# passing priors between phases and using previous phases to modify the image that is fitted. The idea behind this
# pipeline is simple. Basically, if we we have an image of a strong lens where the lens galaxy's light is visible we
# fit it using 3 phases:

# Phase 1) We fit only the lens galaxy's light.
# Phase 2) We fit the lens galaxy's mass and source galaxy's light, using the lens light subtracted image from Phase 1.
# Phase 3) We fit the lens galaxy's light, mass and source galaxy's light, using the results of Phase's 1 and 2 to
#          initialize our non-linear search.

# By breaking the analysis down in this way we achieve much faster lens modeling and avoid inferring an incorrect
# lens model.

# Lets load some simulated lens dataset (which now includes the lens galaxy's light) and fit it using this pipeline. To do
# this, we won't use this Juypter notebook! Instead, go to the script
# 'autolens_workspace/runners/simple/lens_sersic_sie__source_sersic.py'. This runner script does everything
# we need to set off the pipeline, in particular, it:

# 1) Loads the imaging dataset from .fits files.
# 2) Loads the mask of this example dataset from a .fits file.
# 3) Imports and creates the pipeline.
# 4) Uses this pipeline to fit the dataset.

# The results of this pipeline will appear in the 'output' folder of the autolens autolens_workspace. It should take half an
# hour or so to run from start to end. Of course, you can check out the results on-the-fly in the output folder.
