# %%
"""
__SLaM (Source, Light and Mass)__

Welcome to the SLaM pipeline runner, which loads a strong lens dataset and analyses it using a SLaM lens modeling 
pipeline. For a complete description of SLaM, checkout ? and ?.

__THIS RUNNER__

Using two source pipelines and a mass pipeline we will fit a lens model where: 

    - The lens galaxy's _LightProfile_ is fitted with an _EllipticalSersic_.
    - The lens galaxy's _MassProfile_ is fitted with an _EllipticalIsothermal_.
    - The source galaxy is fitted with an *Inversion*.

We'll use the SLaM pipelines:

    'slam/no_lens_light/source/parametric/lens_sie__source_sersic.py'.
    'slam/no_lens_light/source/inversion/from_parametric/lens_sie__source_inversion.py'.
    'slam/no_lens_light/mass/power_law/lens_power_law__source.py'.

Check them out now for a detailed description of the analysis!
"""

# %%
""" AUTOFIT + CONFIG SETUP """

# %%
from autoconf import conf
import autofit as af

# %%
# %%
"""Setup the path to the autolens workspace, using the project pyprojroot which determines it automatically."""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# %%
"""Use this path to explicitly set the config path and output path."""
conf.instance = conf.Config(
    config_path=f"{workspace_path}/config", output_path=f"{workspace_path}/output"
)

# %%
""" AUTOLENS + DATA SETUP """

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "lens_sie__source_sersic"
pixel_scales = 0.1

# %%
"""
Create the path where the dataset will be loaded from, which in this case is
'/autolens_workspace/dataset/imaging/lens_bulge_disk_mlr_nfw__source_sersic'
"""

# %%
dataset_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_type, dataset_label, dataset_name]
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

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

settings = al.PhaseSettingsImaging(
    grid_class=al.Grid,
    sub_size=2,
    grid_inversion_class=al.GridInterpolate,
    pixel_scales_interp=0.1,
    inversion_pixel_limit=1500,
)


# %%
"""
__PIPELINE SETUP__

Pipelines and hyper pipelines used the _PipelineSetup_ object to customize the analysis performed by the pipeline,
for example if a shear was included in the mass model and the model used for the source galaxy.

SLaM pipelines break the analysis down into multiple pipelines which focus on modeling a specifc aspect of the strong 
lens, first the Source, then the (lens) Light and finally the Mass. Each of these pipelines has it own setup object 
which is equivalent to the _PipelineSetup_ object, customizing the analysis in that pipeline.

The setup used in earlier pipelines determine the model used in later pipelines. For example, if the _Source_ pipeline 
is given a _Pixelization_ and _Regularization_, than this _Inversion_ will be used in the subsquent Light and Mass 
pipelines. The assumptions regarding the lens light chosen by the _Light_ object are carried forward to the _Mass_ 
pipeline.

The setup also tags the path structure of every pipeline in a unique way, such than combinations of different
SLaM pipelines can be used to fit lenses with different models. If the earlier pipelines are identical (e.g. they use
the same _SourceSetup_ they will reuse those results before branching off to fit different models in the Light and / or
Mass pipelines. 
"""

# %%
"""
__HYPER SETUP__

The _HyperSetup_ determines which hyper-mode features are used during the model-fit and behaves analgous to the 
hyper inputs of the _PipelineSetup_ object.

The _HyperSetup_ object has a new input available, 'hyper_fixed_after_source', which fixes the hyper-parameters to
the values computed by the hyper-phase at the end of the Source pipeline. By fixing the hyper-parameter values in the
Light and Mass pipelines, model comparison can be performed in a consistent fashion.
"""

# %%
hyper = al.slam.HyperSetup(
    hyper_galaxies=True, hyper_image_sky=False, hyper_background_noise=False
)

# %%
"""
__SourceSetup__

The _SourceSetup_ determines the model-fit used by the Source pipelines. A full description of all options can be 
found ? and ?.

The Source pipeline aims to initialize a robust model for the source galaxy. To do this, it assumes an 
_EllipticalIsothermal_ profile for the lens galaxy's mass and an _EllipticalSersic_ + _EllipticalExponential_ model 
for the lens galaxy's light. Our experience with lens modeling has shown they are the simpliest models that provide a 
good fit to the majority of strong lenses.

For this runner the _SourceSetup_ customizes:

    - The Pixelization used by the inversion of this pipeline.
    - The Regularization scheme used by of this pipeline.
    - If there is an external shear in the mass model or not.

The _SourceSetup_ determines the source model used in the _Light_ and _Mass_ pipelines, which will thus both use an
_Inversion_. If an external shear is omitted from the Source pipeline it can be introduced in the Mass pipeline.
"""

source = al.slam.SourceSetup(
    pixelization=al.pix.VoronoiBrightnessImage, regularization=al.reg.AdaptiveBrightness
)

# %%
"""
__MassSetup__

The Mass pipeline fits the model for the lens galaxy's mass. A full description of all options can be found ? and ?.

The model used to represent the lens galaxy's mass is determined by the pipeline that is imported and made later in 
the script. For this runner an sie is used, which models the lens galaxy's mass as an _EllipticalPowerLaw_.

For this runner the _MassSetup_ customizes:

    - If there is an external shear in the mass model or not.

Certain _MassSetup_ inputs correspond to certain pipelines, for example the 'aligh_bulge_dark_centre'
input is only relevent for Mass pipelines that follow 'bulge_disk' Light pipelines and which use a 'light_dark'
pipeline.
"""

mass = al.slam.MassSetup(no_shear=False)

# %%
"""
__SLaM__

We combine all off the above Setup objects in a _SLaM_ object, which we pass to the pipelines when we make them.

The _SLaM_ object contains a number of methods using in the make_pipeline functions which are used to compose the model 
based on the input Setup values. It also handles pipeline tagging and path structure.
"""

slam = al.slam.SLaM(
    hyper=hyper, source=source, mass=mass, folders=["slam", dataset_type]
)

# %%
"""
__PIPELINE CREATION__

We import and make pipelines as per usual, albeit we'll now be doing this for multiple pipelines!
# %%
"""

# %%
from autolens_workspace.advanced.slam.pipelines.no_lens_light.source.parametric import (
    lens_sie__source_sersic,
)
from autolens_workspace.advanced.slam.pipelines.no_lens_light.source.inversion.from_parametric import (
    lens_sie__source_inversion,
)

source__parametric = lens_sie__source_sersic.make_pipeline(slam=slam, settings=settings)

source__inversion = lens_sie__source_inversion.make_pipeline(
    slam=slam, settings=settings
)

from autolens_workspace.advanced.slam.pipelines.no_lens_light.mass.power_law import (
    lens_power_law__source,
)

mass__power_law = lens_power_law__source.make_pipeline(slam=slam, settings=settings)

# %%
"""
__PIPELINE COMPOSITION AND RUN__

We finally add the pipelines above together, meaning they will run back-to-back, passing information from earlier 
phases to later phases.
"""

pipeline = source__parametric + source__inversion + mass__power_law

pipeline.run(dataset=imaging, mask=mask)
