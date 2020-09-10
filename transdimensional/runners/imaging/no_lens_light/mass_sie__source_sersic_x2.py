# %%
"""
__WELCOME__ 

This transdimensional pipeline runner loads a strong lens dataset and analyses it using a transdimensional lens
modeling pipeline.

Using a pipeline composed of two phases this runner fits _Imaging_ of a strong lens system, where in the final phase
of the pipeline:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's _MassProfile_ is modeled as an _EllipticalIsothermal_.
 - The source galaxy's two _LightProfile_'s are modeled as _EllipticalSersic_'s.

This uses the pipeline (Check it out full description of the pipeline):
"""

# %%
"""Setup the path to the autolens workspace, using pyprojroot to determine it automatically."""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# %%
"""Set up the config and output paths."""

# %%
from autoconf import conf

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
"""Use this path to explicitly set the config path and output path."""

# %%
from autoconf import conf

conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "mass_sie__source_sersic_x2"
pixel_scales = 0.1

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/mass_sie__source_sersic'
"""

# %%
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

# %%
"""Using the dataset path, load the data (image, noise-map, PSF) as an _Imaging_ object from .fits files."""

# %%
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

# %%
"""Next, we create the mask we'll fit this data-set with."""

# %%
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.4
)

# %%
"""Make a quick subplot to make sure the data looks as we expect."""

# %%
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Settings__

The _SettingsPhaseImaging_ describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the 'autolens_workspace/examples/model' example scripts, with a 
complete description of all settings given in 'autolens_workspace/examples/model/customize/settings.py'.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__Pipeline_Setup__:

Pipelines can contain _Setup_ objects, which customize how different aspects of the model are fitted. 

First, we create a a _SetupMassTotal_, which customizes:

 - If there is an _ExternalShear_ in the mass model or not.
"""

# %%
setup_mass = al.SetupMassTotal(no_shear=False)

# %%
"""
Next, we create a _SetupSourceSersic_ which does not customize the pipeline behaviour except for tagging (see below).
"""

# %%
setup_source = al.SetupSourceSersic()

"""
_Pipeline Tagging_

The _Setup_ objects are input into a _SetupPipeline_ object, which is passed into the pipeline and used to customize
the analysis depending on the setup. This includes tagging the output path of a pipeline. For example, if 'no_shear' 
is True, the pipeline's output paths are 'tagged' with the string 'no_shear'.

This means you can run the same pipeline on the same data twice (with and without shear) and the results will go
to different output folders and thus not clash with one another!

The 'folders' below specify the path the pipeline results are written to, which is:

 'autolens_workspace/output/pipelines/dataset_type/dataset_name' 
 'autolens_workspace/output/pipelines/imaging/mass_sie__source_sersic_x2/'
 
 The redshift of the lens and source galaxies are also input (see 'examples/model/customimze/redshift.py') for a 
description of what inputting redshifts into **PyAutoLens** does.
"""

# %%
setup = al.SetupPipeline(
    folders=["transdimensional", f"{dataset_type}_{dataset_label}", dataset_name],
    redshift_lens=0.5,
    redshift_source=1.0,
    setup_mass=setup_mass,
    setup_source=setup_source,
)

# %%
"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its 'make_pipeline' function, inputting the 
*Setup* and *SettingsPhase* above.
"""

# %%
from autolens_workspace.transdimensional.pipelines.imaging.no_lens_light import (
    mass_sie__source_sersic_x2,
)

pipeline = mass_sie__source_sersic_x2.make_pipeline(setup=setup, settings=settings)

# %%
"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

# %%
pipeline.run(dataset=imaging, mask=mask)
