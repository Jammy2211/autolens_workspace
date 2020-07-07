# %%
"""
This example demonstrates how to use interpolated deflection-angles interpolation in the phase settings, which
computes the deflection angles of a mass profile on a coarse lower resolution 'interpolation grid' and interpolates
these values to the image's native sub-grid resolution.

The benefits of this are:

    - For _MassProfile_'s that require computationally expensive numerical integration, this reduces the number of
      integratals performed 100000's to 1000's, giving a potential speed up in run time of x100 or more!

The downsides of this are:

    - The interpolated deflection angles will be inaccurate to some level of precision, depending on the resolution
      of the interpolation grid. This could lead to inaccurate and biased mass models.

The interpolation grid is defined in terms of a pixel scale and it is automatically matched to the mask used in that
phase. A higher resolution grid (i.e. lower pixel scale) will give more precise deflection angles, at the expense
of longer calculation times. In this example we will use an interpolation pixel scale of 0.05", which balances run-time
and precision.

In this example, we will fit the lens galaxy's mass using an _EllipticalSersic_ + _SphericalNFW_ mass model (which
represents the stellar and dark matter of a galaxy). The _EllipticalSersic_ requires expensive numerical intergration,
whereas the _SphericalNFW_ does not. PyAutoLens will only used interpolation for the _EllipticalSersic_, given we can
compute the deflection angles of the _SphericalNFW_ efficiently.

Whether the interpolatioon grid is used for a given mass profile is set in the following config file:

    'autolens_workspace/config/grids/interpolate.ini'

The True and False values reflect whether interpolation is used for each function of each mass profile. The default
values supplied with the autolens_workspace reflect whether the profile requires numerical integration or not.

I'll assume that you are familiar with the beginner example scripts work, so if any code doesn't make sense familiarize
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

Next, we specify the *PhaseSettingsImaging*, which describe how the model is fitted to the data in the log likelihood
function. In this example, we specify:

    - The grid_class as a _GridInterpolate_, telling PyAutoLens to use interpolation when calculation deflection 
      angles.
      
    - A pixel_scales_interp of 0.05, which is the resolution of the interpolation on which the deflection angles are
      computed and used to interpolate to the data's native resolution.   
      
"""

# %%
settings = al.PhaseSettingsImaging(
    grid_class=al.GridInterpolate, pixel_scales_interp=0.05
)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

    '/autolens_workspace/output/examples/settings/lens_sie__source_sersic/phase__interpolation'.

However, because the _PhaseSettings_ include a grid_class and pixel_scales_interp, the output path is tagged to 
reflelct this, meaning the full output path is:

    '/autolens_workspace/output/examples/settings/lens_sie__source_sersic/phase__binned_up/settings__grid_interp_0.05'.

"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__interpolation",
    folders=["examples", "settings"],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

phase.run(dataset=imaging, mask=mask)
