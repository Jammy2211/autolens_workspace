"""
Plots: Publication Subhalo Scan
===============================

Scientific papers require plots with large labels, clear axis ticks and minimal white space.

This example shows how to plot a subhalo scanning map overlaid on an image-plane image.
The map shows the increase in Bayesian evidence of models including a dark matter subhalo.

We do not perform a model-fit to set up the evidence values and instead use a manually-defined
example for efficiency.

For array overlays not supported by the simple `aplt.plot_array()` kwargs, we use matplotlib
directly after setting up the figure.

__Start Here Notebook__

If any code in this script is unclear, refer to `plot/start_here.ipynb`.
"""

from autoconf import jax_wrapper  # Sets JAX environment before other imports

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
import autolens as al
import autolens.plot as aplt

"""
__Dataset__

Load example imaging of a strong lens.
"""
dataset_name = "simple__no_lens_light"
dataset_path = Path("dataset") / "imaging" / dataset_name

dataset = al.Imaging.from_fits(
    data_path=dataset_path / "data.fits",
    psf_path=dataset_path / "psf.fits",
    noise_map_path=dataset_path / "noise_map.fits",
    pixel_scales=0.1,
)

mask = al.Mask2D.circular(
    radius=3.0, shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales
)

dataset = dataset.apply_mask(mask=mask)

"""
__Base Image__

Plot the data alone first to see the baseline appearance.
"""
aplt.plot_array(array=dataset.data, title="Data")

"""
__Subhalo Evidence Array__

Define a 5x5 example grid of Bayesian evidence increases, as if returned by a subhalo scan.
Each value is the log evidence increase when including a subhalo at that grid position.

The tutorial `autolens_workspace/results/advanced/result_subhalo_grid.py` shows how to compute
this from a real model-fit.
"""
evidence_array = al.Array2D.no_mask(
    values=[
        [-1.0, -1.0, 1.0, 2.0, 0.0],
        [0.0, 1.0, 2.0, 3.0, 2.0],
        [0.0, 3.0, 5.0, 20.0, 8.0],
        [0.0, 1.0, 1.0, 15.0, 5.0],
        [0.0, 1.0, 1.0, 0.0, 0.0],
    ],
    pixel_scales=1.0,
)

evidence_max = float(np.max(evidence_array))
evidence_half = evidence_max / 2.0

"""
__Subhalo Overlay__

We now overlay the evidence array on the data image using matplotlib directly.

The workflow is:

 1. Plot the base data image with `aplt.plot_array()`.
 2. Use `plt.gca()` to get the current axes.
 3. Use `ax.imshow()` to overlay the evidence array with a colorful colormap and transparency.
 4. Add a colorbar for the evidence overlay.
 5. Apply publication-quality axis tick settings.
 6. Show / save the figure.

The evidence array is resampled to cover the same spatial extent as the data image.
"""
aplt.plot_array(array=dataset.data, title="Subhalo Scanning Map")

ax = plt.gca()
fig = plt.gcf()

# Overlay the evidence array: extent should match the data image extent
data_extent = dataset.data.shape_native
half_arcsec = (data_extent[0] / 2) * dataset.data.pixel_scales[0]
extent = [-half_arcsec, half_arcsec, -half_arcsec, half_arcsec]

evidence_clipped = np.clip(evidence_array.native, 0.0, evidence_max)
ax.imshow(
    evidence_clipped,
    cmap="RdYlGn",
    alpha=0.6,
    vmin=0.0,
    vmax=evidence_max,
    extent=extent,
    origin="upper",
    aspect="auto",
)

# Add a colorbar for the evidence overlay
sm = cm.ScalarMappable(
    cmap="RdYlGn",
    norm=mcolors.Normalize(vmin=0.0, vmax=evidence_max),
)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_ticks([0.0, evidence_half, evidence_max])
cbar.set_ticklabels(
    [f"{0.0:.1f}", f"{evidence_half:.1f}", f"{evidence_max:.1f}"],
    fontsize=18,
)
cbar.ax.tick_params(labelrotation=90)

# Publication-quality tick settings
tick_values = [-2.5, 0.0, 2.5]
ax.set_xticks(tick_values)
ax.set_yticks(tick_values)
ax.set_xticklabels([f'{v}"' for v in tick_values], fontsize=22)
ax.set_yticklabels(
    [f'{v}"' for v in tick_values], fontsize=22, rotation="vertical", va="center"
)
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.show()
plt.close()

"""
Notes on the settings used above:

 - `cmap="RdYlGn"`: A diverging red-yellow-green colormap makes low-evidence regions (near zero)
   and high-evidence detections visually distinct from the underlying image colormap.

 - `alpha=0.6`: Makes the overlay translucent so the underlying data image remains visible.

 - `vmin=0.0, vmax=evidence_max`: The colorbar spans from zero to the maximum evidence value,
   making it easy to read off the subhalo detection significance.
"""
