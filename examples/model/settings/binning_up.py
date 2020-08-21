# %%
"""
This example demonstrates how to use the binning up in the phase settings, where binning up fits a lower resolution
binned up version of the dataset.

The benefits of this are:

 - It can give significant gains in computational run-times.

The drawbacks of this are:

 - The lower resolutioon data will constrain the lens model worse, giving larger errors or a biased model.

 - Binning up the Point Spread Function of the dataset will less accurately represent the optics of the observation,
      again leading to inaccurate lens models with larger errors.

I'll assume that you are familiar with the beginner example scripts, so if any code doesn't make sense familiarize
yourself with those first!
"""

# %%
"""Setup the path to the autolens workspace, using the project pyprojroot which determines it automatically."""

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
""" AUTOLENS + DATA SETUP """

# %%
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "lens_sie__source_sersic"
pixel_scales = 0.1

dataset_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_type, dataset_label, dataset_name]
)

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

# %%
"""
__Model__

We'll fit a _EllipticalIsothermal + _EllipticalSersic_ model which we often fitted in the beginner example scripts.
"""

# %%
lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, light=al.lp.EllipticalSersic)

# %%
"""
__Search__

We'll use the default DynestyStatic sampler we used in the beginner examples.
"""

# %%
search = af.DynestyStatic(n_live_points=50)

# %%
"""
__Settings__

Next, we specify the *SettingsPhaseImaging*, which describe how the model is fitted to the data in the log likelihood
function. In this example, we specify:
 
 - A bin_up_factor of 2, meaning the dataset is binned up from a resolution of 0.1" per pixel to a resolution 
      of 0.2" per pixel before we perform the model-fit.
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(bin_up_factor=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/settings/lens_sie__source_sersic/phase__binned_up'.
    
However, because the _SettingsPhase_ include a bin_up_factor, the output path is tagged to reflelct this, meaning the
full output path is:

 '/autolens_workspace/output/examples/settings/lens_sie__source_sersic/phase__binned_up/settings__bin_up_2'.

"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__binned_up",
    folders=["examples", "settings"],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

phase.run(dataset=imaging, mask=mask)
