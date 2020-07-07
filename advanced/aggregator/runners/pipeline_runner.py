"""
__Aggregator: Pipeline Runner__

This script fits the sample of three strong lenses simulated by the script 'autolens_workspace/aggregator/sample.py'
using a pipeline to illustrate aggregator functionality in the tutorial:

    - a5_pipelines
    - a6_advanced
 
If you are not yet familiar with PyAutoLens's pipeline functionality, you should checkout 
'autolens_workspace/pipelines' and 'howtolens/chapter_3_pipelines' before doing these tutorials.

Using a pipeline composed of three phases this runner fits imaging of a strong lens system, where: 
 
    - An _EllipticalIsothermal_ _MassProfile_ for the lens galaxy's mass.
    - An _Inversion_ for the source galaxy's light.
"""

""" AUTOFIT + CONFIG SETUP """

from autoconf import conf
import autofit as af

# %%
"""Setup the path to the autolens workspace, using the project pyprojroot which determines it automatically."""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

"""Use this path to explicitly set the config path and output path."""
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

""" AUTOLENS + DATA SETUP """

import autolens as al

"""Specify the dataset type, label and name, which we use to determine the path we load the data from."""

pixel_scales = 0.1

for dataset_name in [
    "lens_sie__source_sersic__0",
    "lens_sie__source_sersic__1",
    "lens_sie__source_sersic__2",
]:

    """Set up the config and output paths."""
    dataset_path = af.util.create_path(
        path=workspace_path, folders=["dataset", "aggregator", dataset_name]
    )

    """Using the dataset path, load the data (image, noise-map, PSF) as an imaging object from .fits files."""
    imaging = al.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
    )

    """Next, we create the mask we'll fit this data-set with."""
    mask = al.Mask.circular(
        shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
    )

    # %%
    """
    The _PhaseSettings_ (which customize the fit of the phase's fit), will also be available to the aggregator!
    """

    # %%
    settings = al.PhaseSettingsImaging(grid_class=al.Grid, sub_size=2)

    # %%
    """
    __Pipeline_Setup_And_Tagging__:

    For this pipeline the pipeline setup customizes:

        - The Pixelization used by the inversion of this pipeline.
        - The Regularization scheme used by of this pipeline.
        - If there is an external shear in the mass model or not.
    """

    setup = al.PipelineSetup(
        pixelization=al.pix.VoronoiMagnification,
        regularization=al.reg.Constant,
        no_shear=True,
        folders=["aggregator", dataset_name],
    )

    from pipelines.imaging.no_lens_light import lens_sie__source_inversion

    pipeline = lens_sie__source_inversion.make_pipeline(setup=setup, settings=settings)

    pipeline.run(dataset=imaging, mask=mask)
