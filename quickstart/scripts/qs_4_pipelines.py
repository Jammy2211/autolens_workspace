# %%
"""
__PIPELINES__

Above, we used the phase module to create a phase, which fitted a lens and source galaxy model to some Imaging data
using MultiNest. In PyAutoLens, a pipeline is effectively a series of phases that are linked together, which
ultimately fit a lens model. Why do we break the analysis down into multiple phases, instead of using just 1 phase?

For highly complex lens models, fitting all lens model parameters without any initialization will lead to either
extremely slow and inefficient lens modeling or, worse still, cause us to infer an incorrect lens model. Pipelines
break the fit down by initially fitting simple lens models, and then gradually make the model more complex, using the
results of the earlier phases to initialize the non-linear sampler.

Lets look at an example. Go to 'autolens_workspace/pipelines/beginner/lens_sersic_sie_shear_source_sersic.py' First,
have a skim read of the pipeline - you'll see it uses phases like the one we made above, but introduces a number of
other concepts (e.g. prior passing, image modification) that we haven't covered in the quick-start tutorial.

Nevertheless, the main point of this pipeline is straight forward. We assume we have an image of a strong lens where
the lens galaxy's light is also visible, and we fitting it using 3 phases:

Phase 1) We fit only the lens galaxy's light.
Phase 2) We fit the lens galaxy's mass and source galaxy's light, using the lens light subtracted image from Phase 1.
Phase 3) We fit the lens galaxy's light and mass and source galaxy's light, using the results of Phase's 1 and 2 to
         initialize our non-linear search.

By breaking the analysis down in this way, we achieve much faster lens modeling and will avoid inferring an
incorrect lens model.

Lets load some simulated lens data (which now includes the lens galaxy's light) and fit it using this pipeline. To
do this, we won't use this Juypter notebook! Instead, go to the script
'autolens_workspace/runners/beginner/runner_lens_light_mass_and_source.py'. This runner script does everything we need
to set off the pipeline, in particular, it:

1) Loads the imaging data from .fits files.
2) Loads the mask of this example data from a .fits file.
3) Imports and creates the pipeline.
4) Uses this pipeline to fit the data.

The results of this pipeline will appear in the 'output' folder of the autolens workspace. It should take half an
hour or so to run from start to end. Of course, you can check out the results on-the-fly in the output folder.

"""
