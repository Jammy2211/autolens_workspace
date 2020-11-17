"""
This example demonstrates how to change the sub-gridding of the phase settings, which changes the resolution of the
sub-grid that oversamples the `LightProfile` intensities and `MassProfile` deflection angle calculations.

The benefits of this are:

 - A higher level of sub-gridding provides numerically more precise results.

The drawbacks of this are:

-   Longer calculations and higher memory usage.

I`ll assume that you are familiar with the beginner example scripts work, so if any code doesn`t make sense familiarize
yourself with those first!
"""

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__source_sersic"
pixel_scales = 0.1

dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)

imaging = al.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
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

 - A sub_size of 4, meaning we use a high resolution 4x4 sub-grid instead of the default 2x2 sub-grid.
"""

settings_masked_imaging = al.SettingsMaskedImaging(grid_class=al.Grid, sub_size=4)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__Search__

we'll use the default `DynestyStatic` sampler we used in the beginner examples.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase_sub`.

However, because the `SettingsPhase` include a bin_up_factor, the output path is tagged to reflelct this, meaning the
full output path is:

 `/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase_sub/settings__grid_sub_4`.
"""

search = af.DynestyStatic(
    path_prefix=path.join("examples", "settings"), name="phase_sub", n_live_points=50
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
