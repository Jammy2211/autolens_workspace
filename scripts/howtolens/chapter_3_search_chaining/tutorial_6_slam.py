"""
Tutorial 6: SLaM
================

You are now familiar with pipelines, in particular how we use them to break-down the lens modeling procedure
to provide more efficient and reliable model-fits. In the previous tutorials, you learnt how to write your own
pipelines, which can fit whatever lens model is of particular interest to your scientific study.

However, for most lens models there are standardized approaches one can take to fitting them. For example, as we saw in
tutorial 3 of this chapter, an effective approach is to fit a model for the lens's light followed by a model for its
mass and the source. It would be wasteful for all **PyAutoLens** users to have to write their own pipelines to
perform the same tasks.

For this reason, the `autolens_workspace` comes with a number of standardized pipelines, which fit common lens models
in ways we have tested are efficient and robust. These pipelines can be found the `chaining/pipelines` packages and
can be easily adapted to fit lens models with a range of different parameterizations and assumptions.

__SLaM (Source, Light and Mass)__

A second set of template pipelines, called the **SLaM** (Source, Light and Mass) pipelines can be found in the folder
`autolens_workspace/slam`. These are similar in design to the pipelines, but are composed of a specific sequence of
pipelines:

 - `Source`: A pipeline that focuses on producing a robust model for the source's light, using simpler models for the 
 lens's light and mass.
   
 - `Light`: A pipeline that fits a complex lens light model (e.g. one with many components), using the initialized 
 source model to cleanly deblend the lens and source light.
   
 - `Mass`: A pipeline that fits a complex lens mass model, benefitting from the good models for the lens's light and 
 source.

For fitting very complex lens models, for example ones which decompose its mass into its stellar and dark components,
the **SLaM** pipelines have been carefully crafted to do this in a reliable and automated way that is still efficient.
The **SLaM** pipelines also make fitting many different models to a single dataset efficient, as they reuse the results
of earlier searches (e.g. in the Source pipeline) to fit different models in the `Light` and `Mass` pipelines for the 
lens's  light and mass.

Whether you should use individual searches, pipelines, The SLaM pipelines or write your own model-fitting script
depends on the scope  of your scientific analysis. I would advise you begin by trying to adapting the scripts in the
`autolens_workspace`to fit your data, and also try using the SLaM pipelines once you are a confident **PyAutoLens** user.
"""
