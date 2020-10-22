# %%
"""
Tutorial 4: Setup and SLaM
==========================

You are now familiar with pipelines, in particular how we use them to break-down the lens modeling procedure
to provide more efficient and reliable model-fits. In the previous tutorials, you've learnt how to write
your own pipelines, which can fit whatever lens model is of particular interest to your scientific study.

However, for most lens models there are standardized approaches one can take to fitting them. For example, as we saw in
tutorial 1 of this chapter, it is an effective approch to fit a model for the lens's light followed by a model for its
mass and the source. It would be wasteful for all **PyAutoLens** users to have to write their own pipelines to
perform the same tasks.

For this reason, the `autolens_workspace` comes with a number of standardized pipelines, which fit common lens models
in ways we have tested are efficient and robust. These pipelines also use `Setup` objects to customize the creating of
the lens and source `PriorModel`'s, making it straight forward to use the same pipeline to fit a range of different
lens model parameterizations.

Lets take a look.
"""

# %%
#%matplotlib inline

from pyprojroot import here

workspace_path = str(here())
#%cd $workspace_path
print(f"Working Directory has been set to `{workspace_path}`")

import autolens as al
import autolens.plot as aplt

# %%
"""
Lets begin with the `SetupLightParametric` object, which describes how we set up our parametric model using 
`LightProfile`'s for the lens's light (the lens's light cannot be fitted with an `Inversion`, but we nevertheless
call it parametric to make it clear the model uses `LightProfile`'s.

This object customizes:

 - The `LightProfile`'s use to fit different components of the lens light, such as its `bulge` and `disk`.
 - The alignment of these components, for example if the `bulge` and `disk` centres are aligned.
 - If the centre of the lens light profile is manually input and fixed for modeling.
"""

# %%
setup_light = al.SetupLightParametric(
    bulge_prior_model=al.lp.EllipticalSersic,
    disk_prior_model=al.lp.EllipticalSersic,
    envelope_prior_model=None,
    align_bulge_disk_centre=True,
    align_bulge_disk_elliptical_comps=False,
    light_centre=None,
)

# %%
"""
In the `Setup` above we made the lens's `bulge` and `disk` use the `EllipticalSersic` `LightProfile`, which we
can verify below:
"""

# %%
print(setup_light.bulge_prior_model)
print(setup_light.bulge_prior_model.cls)
print(setup_light.disk_prior_model)
print(setup_light.disk_prior_model.cls)

# %%
"""
We can also verify that the `bulge` and `disk` share the same prior on the centre because we aligned them
by setting `align_bulge_disk_centre=True`:
"""

# %%
print(setup_light.bulge_prior_model.centre)
print(setup_light.disk_prior_model.centre)

# %%
"""
When `GalaxyModel`'s are created in the template pipelines in the `autolens_workspace/transdimensional/pipelines`
and `autolens_workspace/slam/pipelines` they use the `bulge_prior_model`, `disk_prior_model`, etc to create them (as 
opposed to explcitly writing the classes in the pipelines, as we did in the previous tutorials).
"""
