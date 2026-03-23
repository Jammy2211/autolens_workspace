"""
Plots: Multi Subplots
=====================

This example shows how to combine figures from different PyAutoLens objects onto a single
matplotlib subplot.

In the old API, this was done using `open_subplot_figure()` / `close_subplot_figure()` methods
on plotter objects. These methods no longer exist.

In the new API, we use matplotlib directly to create a subplot grid and call `aplt.plot_array()`
or `aplt.subplot_*()` functions to fill each panel.

The example below combines an imaging dataset's data and signal-to-noise map with a fit's
normalized residual map and chi-squared map — the same combination shown in the old API example.

__Start Here Notebook__

If any code in this script is unclear, refer to `plot/start_here.ipynb`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import matplotlib.pyplot as plt
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load and mask the `lens_sersic` dataset.
"""
dataset_name = "lens_sersic"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask_radius = 3.0

mask = al.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

dataset = dataset.apply_mask(mask=mask)

lens_galaxy = al.Galaxy(
    redshift=0.5,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
    ),
    shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        intensity=4.0,
        effective_radius=0.1,
        sersic_index=1.0,
    ),
)

tracer = al.Tracer(galaxies=[lens_galaxy, source_galaxy])

fit = al.FitImaging(dataset=dataset, tracer=tracer)

"""
__Multi Subplot__

We create a 1x5 subplot using matplotlib, then fill each panel by calling `aplt.plot_array()`
with `output_path` pointing to a temporary file and loading the result — or, more simply,
by plotting each array into its own axes.

The approach below uses matplotlib's `add_subplot` API directly. We set the current axes
using `plt.sca()` before each call so that `aplt.plot_array()` draws into the correct panel.

Note: `aplt.plot_array()` calls `plt.show()` or saves internally. For custom subplots,
the simplest approach is to use `imshow()` directly on each matplotlib axes object.
"""
arrays = [
    (dataset.data, "Data"),
    (dataset.signal_to_noise_map, "Signal-to-Noise Map"),
    (fit.model_image, "Model Image"),
    (fit.normalized_residual_map, "Normalized Residuals"),
    (fit.chi_squared_map, "Chi-Squared Map"),
]

fig, axes = plt.subplots(1, 5, figsize=(18, 3))

for ax, (array, title) in zip(axes, arrays):
    im = ax.imshow(array.native, origin="upper", cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("subplot_multi.png", bbox_inches="tight")
plt.show()
plt.close()

"""
__Subplot with Dedicated subplot_* Functions__

Alternatively, the dedicated `aplt.subplot_imaging_dataset()` and `aplt.subplot_fit_imaging()`
functions produce complete multi-panel overviews automatically and are the preferred way to
visualize these objects.
"""
aplt.subplot_imaging_dataset(dataset=dataset)
aplt.subplot_fit_imaging(fit=fit)
