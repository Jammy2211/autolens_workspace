import os

# %%
"""
__WELCOME__ 

Welcome to the pipeline runner, which loads a strong lens dataset and analyses it using a lens modeling pipeline.

Using a pipeline composed of three phases this runner fits imaging of a strong lens system, where: 

    - The lens galaxy's light is omitted from the data and model.
    - The lens galaxy's _MassProfile_ is fitted with an _EllipticalIsothermal_.
    - The source galaxy is fitted with an *Inversion*.

This uses the pipeline (Check it out full description of the pipeline):

    'autolens_workspace/pipelines/imaging/no_lens_light/lens_sie__source_inversion.py'.
"""

# %%
""" AUTOFIT + CONFIG SETUP """

# %%
from autoconf import conf
import autofit as af

# %%
"""Setup the path to the autolens_workspace, using a relative directory name."""

# %%
workspace_path = "{}/../../..".format(os.path.dirname(os.path.realpath(__file__)))

# %%
"""Use this path to explicitly set the config path and output path."""

# %%
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autolens as al
import autolens.plot as aplt

# %%
"""Specify the dataset label and name, which we use to determine the path we load the data from."""

# %%
dataset_label = "imaging"
dataset_name = "lens_sie__source_sersic"

# %%
"""This is the pixel-to-arcsecond conversion factor of the data, you must get this right!"""

# %%
pixel_scales = 0.1

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/lens_sie__source_sersic/'
"""

# %%
dataset_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_label, dataset_name]
)

# %%
"""Using the dataset path, load the data (image, noise-map, PSF) as an imaging object from .fits files."""

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
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

# %%
"""Make a quick subplot to make sure the data looks as we expect."""

# %%
aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Settings__

The *PhaseSettingsImaging* describe how the model is fitted to the data in the log likelihood function.

These settings are used and described throughout the 'autolens_workspace/examples/model' example scripts, with a 
complete description of all settings given in 'autolens_workspace/examples/model/customize/settings.py'.

The settings chosen here are applied to all phases in the pipeline.
"""

# %%
settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=2)

# %%
"""
__Pipeline_Setup_And_Tagging__:

For this runner the _PipelineSetup_ customizes:

    - The Pixelization used by the inversion of this pipeline.
    - The Regularization scheme used by of this pipeline.
    - If there is an external shear in the mass model or not.

The _PipelineSetup_ 'tags' the output path of a pipeline. For example, if 'no_shear' is True, the pipeline's output 
paths are 'tagged' with the string 'no_shear'.

This means you can run the same pipeline on the same data twice (with and without shear) and the results will go
to different output folders and thus not clash with one another!

The 'folders' below specify the path the pipeline results are written to, which is:

    'autolens_workspace/output/dataset_label/dataset_name/' 
    'autolens_workspace/output/imaging/lens_sie__source_sersic/'
"""

# %%
setup = al.PipelineSetup(
    pixelization=al.pix.VoronoiMagnification,
    regularization=al.reg.Constant,
    no_shear=False,
    folders=["pipelines", dataset_label, dataset_name],
)

# %%
"""
__Pipeline Creation__

To create a pipeline we import it from the pipelines folder and run its 'make_pipeline' function, inputting the 
*Setup* and *PhaseSettings* above.
"""

# %%
from pipelines.imaging.no_lens_light import lens_sie__source_inversion

pipeline = lens_sie__source_inversion.make_pipeline(setup=setup, settings=settings)

# %%
"""
__Pipeline Run__

Running a pipeline is the same as running a phase, we simply pass it our lens dataset and mask to its run function.
"""

# %%
pipeline.run(dataset=imaging, mask=mask)
