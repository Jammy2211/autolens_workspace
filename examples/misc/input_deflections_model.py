"""
__Example: Known Deflections Model__

This example is a continuation of the script `input_deflections_sourre_planes.py`. You should read through that
script if you have not done so already before covering this script.

As we discussed, we can now use an input deflection angle map from an external source to create lensed images of
source galaxies. In this example, we assume the source is not known and something we fit for via lens modeling.

To begin, we set up the `InputDeflections` object in an identical fashion to the previous example. The code between the
### -------------- ### is unchanged and can be skipped (you should copy and paste in the code you use to load your own
deflection angle map here once you have run this example).
"""

### --------------------------------------------------------------------------------------------------------------- ###

# %%
"""Use the WORKSPACE environment variable to determine the path to the `autolens_workspace`."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

import autofit as af
import autolens as al
import autolens.plot as aplt

"""
In this example, our `input` deflection angle map is the true deflection angles of the `Imaging` data simulated in the 
`mass_sie__source_sersic.py` simulator. You should be able to simply edit the `from_fits` methods below to point
to your own dataset an deflection maps.

Lets load and plot this dataset.
"""
dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "mass_sie__source_sersic"
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_label}/{dataset_name}"

imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    pixel_scales=0.1,
)

aplt.Imaging.subplot_imaging(imaging=imaging)

"""
In `autolens_workspace/examples/misc/files` you`ll find the script `make_source_plane.py`, which creates the image-plane 
_Grid_ and deflection angles we use in this example (which are identical to those used in the 
`mass_sie__source_sersic.py` simulator). 
"""

"""Lets load the input deflection angle map from a .fits files (which is created in the code mentioned above)."""
deflections_y = al.Array.from_fits(
    file_path=f"{workspace_path}/examples/misc/files/deflections_y.fits",
    pixel_scales=imaging.pixel_scales,
)
deflections_x = al.Array.from_fits(
    file_path=f"{workspace_path}/examples/misc/files/deflections_x.fits",
    pixel_scales=imaging.pixel_scales,
)

"""Lets plot the deflection angles to make sure they look like what we expect!"""
aplt.Array(array=deflections_y)
aplt.Array(array=deflections_x)

"""Lets next load and plot the image-plane grid"""
grid = al.Grid.from_fits(
    file_path=f"{workspace_path}/examples/misc/files/grid.fits",
    pixel_scales=imaging.pixel_scales,
)
aplt.Grid(grid=grid)

### --------------------------------------------------------------------------------------------------------------- ###

"""
The `Mask` our model-fit using the `InputDeflections` will use. This is set up the same way as the previous script, but
not this `Mask` now defines the image-plane region we will fit the data (and therefore where our residuals, chi-squared,
likelihood, etc is calculated.
"""

mask = al.Mask2D.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0, sub_size=2
)

grid = al.Grid.from_mask(mask=mask)

"""
We create the `InputDeflections` `MassProfile`.almosst the same as the previous example. This is going to be passed to 
a  `GalaxyModel` below, so we can use it with a source model to fit to the `Imaging` data using a non-linear search.

However, we pass two additional parameters, `preload_grid` and `preload_blurring_grid`. 

The interpolation performed by the `InputDeflections` can be computationally slow, and if we did it for every 
lens model we fit to the data we`d waste a lot of time. However, because our deflection angle map is fixed and the 
`grid` and `blurring_grid` we interpolated it to are fixed, by passing the latter as a `preload_grid` we can skip
this expensive repeated calculation and speed up the code significantly. Yay!
"""
image_plane_grid = al.Grid.uniform(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1
)
input_deflections = al.mp.InputDeflections(
    deflections_y=deflections_y,
    deflections_x=deflections_x,
    image_plane_grid=image_plane_grid,
    preload_grid=grid,
    preload_blurring_grid=al.Grid.blurring_grid_from_mask_and_kernel_shape(
        mask=mask, kernel_shape_2d=imaging.psf.shape_2d
    ),
)

"""
We now create the lens and source `GalaxyModel``., where the source is an _EllipticalSersic-.
"""
lens = al.GalaxyModel(redshift=0.5, mass=input_deflections)

source = al.GalaxyModel(redshift=1.0, sersic=al.lp.EllipticalSersic)

# %%
"""
__Settings__

Next, we specify the `SettingsPhaseImaging`, which describe how the model is fitted to the data in the log likelihood
function. If you are not familiar with this checkout the example model scripts in `autolens_workspace/examples/model`. 
Below, we specify:

Different *SettingsPhase* are used in different example model scripts and a full description of all *SettingsPhase* 
can be found in the example script `autolens/workspace/examples/model/customize/settings.py` and the following 
link -> <link>

The `preload_grid` input into the `InputDelections` 
"""

# %%
settings_masked_imaging = al.SettingsMaskedImaging(
    grid_class=al.Grid, sub_size=mask.sub_size
)

settings = al.SettingsPhaseImaging(settings_masked_imaging=settings_masked_imaging)

# %%
"""
__Search__

The source is fitted to the `Imaging` data via the input deflection angles using a *NonLinearSearch*, which we 
specify below as the nested sampling algorithm Dynesty (https://dynesty.readthedocs.io/en/latest/). Checkout 
other examples on the workspace if you are unsure what this does!

The script `autolens_workspace/examples/model/customize/non_linear_searches.py` gives a description of the types of
non-linear searches that can be used with **PyAutoLens**. If you do not know what a non-linear search is or how it 
operates, I recommend you complete chapters 1 and 2 of the HowToLens lecture series.
"""

# %%
search = af.DynestyStatic(n_live_points=100)

# %%
"""
__Phase__

We can now combine the model, settings and non-linear search above to create and run a phase, fitting our data with
the lens model.

The phase_name and folders inputs below specify the path of the results in the output folder:  

 `/autolens_workspace/output/examples/beginner/light_sersic__mass_sie__source_sersic/phase__light_sersic__mass_sie__source_sersic`.
"""

# %%
phase = al.PhaseImaging(
    phase_name="phase__input_deflections",
    folders=["misc", dataset_name],
    galaxies=dict(lens=lens, source=source),
    settings=settings,
    search=search,
)

# %%
"""
We can now begin the fit by passing the dataset and mask to the phase, which will use the non-linear search to fit
the model to the data. 

The fit outputs visualization on-the-fly, so checkout the path 
`/path/to/autolens_workspace/output/misc/phase__input_deflections` to see how your fit is doing!
"""

# %%
result = phase.run(dataset=imaging, mask=mask)
