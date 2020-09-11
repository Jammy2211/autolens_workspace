# %%
"""
This example demonstrates how to change the sub-gridding of the phase settings, which changes the resolution of the
sub-grid that oversamples the _LightProfile_ intensities and _MassProfile_ deflection angle calculations.

The benefits of this are:

 - A higher level of sub-gridding provides numerically more precise results.

The drawbacks of this are:

-   Longer calculations and higher memory usage.

I'll assume that you are familiar with the beginner example scripts work, so if any code doesn't make sense familiarize
yourself with those first!
"""

# %%
"""Use the WORKSPACE environment variable to determine the path to the autolens workspace."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
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
dataset_name = "mass_sie__source_sersic"
pixel_scales = 0.1

dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

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
source = al.GalaxyModel(redshift=1.0, sersic=al.lp.EllipticalSersic)

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

Next, we specify the _SettingsPhaseImaging_, which describe how the model is fitted to the data in the log likelihood
function. In this example, we specify:

 - A sub_size of 4, meaning we use a high resolution 4x4 sub-grid instead of the default 2x2 sub-grid.
"""

# %%
settings = al.SettingsPhaseImaging(grid_class=al.Grid, sub_size=4)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 '/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase__sub'.

However, because the _SettingsPhase_ include a bin_up_factor, the output path is tagged to reflelct this, meaning the
full output path is:

 '/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase__sub/settings__grid_sub_4'.

"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__sub",
    folders=["examples", "settings"],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

phase.run(dataset=imaging, mask=mask)
