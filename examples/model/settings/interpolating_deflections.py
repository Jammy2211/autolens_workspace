"""
This example demonstrates how to use interpolated deflection-angles interpolation in the phase settings, which
computes the deflection angles of a `MassProfile` on a coarse lower resolution `interpolation grid` and interpolates
these values to the image`s native sub-grid resolution.

The benefits of this are:

 - For `MassProfile`'s that require computationally expensive numerical integration, this reduces the number of
   integrals performed 100000`s to 1000`s, giving a potential speed up in run time of x100 or more!

The downsides of this are:

 - The interpolated deflection angles will be inaccurate to some level of precision, depending on the resolution
   of the interpolation grid. This could lead to inaccurate and biased mass models.

The interpolation grid is defined in terms of a pixel scale and it is automatically matched to the mask used in that
phase. A higher resolution grid (i.e. lower pixel scale) will give more precise deflection angles, at the expense
of longer calculation times. In this example we will use an interpolation pixel scale of 0.05", which balances run-time
and precision.

In this example, we fit the lens`s `MassProfile`'s using an `EllipticalSersic` + `SphericalNFW` mass model (which
represents the stellar and dark matter of a galaxy). The `EllipticalSersic` requires expensive numerical intergration,
whereas the `SphericalNFW` does not. PyAutoLens will only used interpolation for the `EllipticalSersic`, given we can
compute the deflection angles of the `SphericalNFW` efficiently.

Whether the interpolatioon grid is used for a given `MassProfile` is set in the following config file:

 `autolens_workspace/config/grids/interpolate.ini`

The `True` and `False` values reflect whether interpolation is used for each function of each mass profile. The default
values supplied with the autolens_workspace reflect whether the profile requires numerical integration or not.

I`ll assume that you are familiar with the beginner example scripts work, so if any code doesn`t make sense familiarize
yourself with those first!
"""

import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__source_sersic"
pixel_scales = 0.1

dataset_path = f"dataset/imaging/no_lens_light/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=pixel_scales,
)

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

aplt.Imaging.subplot_imaging(imaging=imaging, mask=mask)

"""
__Model__

we'll fit a `EllipticalIsothermal` + `EllipticalSersic` model which we often fitted in the beginner example scripts.
"""

lens = al.GalaxyModel(redshift=0.5, mass=al.mp.EllipticalIsothermal)
source = al.GalaxyModel(redshift=1.0, bulge=al.lp.EllipticalSersic)

"""
__Settings__

Next, we specify the `SettingsPhaseImaging`, which describe how the model is fitted to the data in the log likelihood
function. In this example, we specify:

 - The grid_class as a `GridInterpolate`, telling PyAutoLens to use interpolation when calculation deflection 
      angles.
      
 - A pixel_scales_interp of 0.05, which is the resolution of the interpolation on which the deflection angles are
      computed and used to interpolate to the data`s native resolution.   
"""

settings_masked_imaging = al.SettingsMaskedImaging(
    grid_class=al.GridInterpolate, pixel_scales_interp=0.05
)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__Search__

we'll use the default `DynestyStatic` sampler we used in the beginner examples.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase_interpolation`.

However, because the `SettingsPhase` include a grid_class and pixel_scales_interp, the output path is tagged to 
reflelct this, meaning the full output path is:

 `/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase_interpolation/settings__grid_interp_0.05`.
"""

search = af.DynestyStatic(
    path_prefix=f"examples/settings", name="phase_interpolation", n_live_points=50
)

"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.
"""

phase = al.PhaseImaging(
    search=search, galaxies=dict(lens=lens, source=source), settings=settings
)

phase.run(dataset=imaging, mask=mask)
