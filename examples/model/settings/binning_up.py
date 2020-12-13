"""
This example demonstrates how to use the binning up in the phase settings, where binning up fits a lower resolution
binned up version of the dataset.

The benefits of this are:

 - It can give significant gains in computational run-times.

The drawbacks of this are:

 - The lower resolutioon data will constrain the lens model worse, giving larger errors or a biased model.

 - Binning up the Point Spread Function of the dataset will less accurately represent the optics of the observation,
      again leading to inaccurate lens models with larger errors.

I`ll assume that you are familiar with the beginner example scripts, so if any code doesn`t make sense familiarize
yourself with those first!
"""

from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

dataset_name = "mass_sie__source_sersic"
dataset_path = path.join("dataset", "imaging", "no_lens_light", dataset_name)
pixel_scales = 0.1

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
 
 - A bin_up_factor of 2, meaning the dataset is binned up from a resolution of 0.1" per pixel to a resolution 
      of 0.2" per pixel before we perform the model-fit.
"""

settings_masked_imaging = al.SettingsMaskedImaging(bin_up_factor=2)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

"""
__Search__

we'll use the default `DynestyStatic` sampler we used in the beginner examples.

The `name` and `path_prefix` below specify the path where results are stored in the output folder:  

 `/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase_binned_up`.

However, because the `SettingsPhase` include a bin_up_factor, the output path is tagged to reflelct this, meaning the
full output path is:

 `/autolens_workspace/output/examples/settings/mass_sie__source_sersic/phase_binned_up/settings__bin_up_2`.
"""

search = af.DynestyStatic(
    path_prefix=path.join("examples", "settings"),
    name="phase_binned_up",
    n_live_points=50,
)

"""
__Phase__

We can now combine the model, settings and `NonLinearSearch` above to create and run a phase, fitting our data with
the lens model.
"""

phase = al.PhaseImaging(
    search=search,
    galaxies=af.CollectionPriorModel(lens=lens, source=source),
    settings=settings,
)

phase.run(dataset=imaging, mask=mask)
