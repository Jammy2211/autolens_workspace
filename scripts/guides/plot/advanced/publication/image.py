"""
Plots: Publication Image
========================

Scientific papers require plots with large labels, clear axis ticks and minimal white space.

This example shows how to produce a publication-quality image-plane plot using the new
plotting API. Customization is done by passing keyword arguments directly to `aplt.plot_array()`.

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
import matplotlib.ticker as ticker
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
__Default Plot__

First, plot the image with default settings so we know what we are starting from.
"""
aplt.plot_array(array=dataset.data, title="Data (Default)")

"""
__Publication Plot__

For a publication, we want:

 - A descriptive title with a large fontsize.
 - Large tick fontsizes.
 - Clean tick values (not the default unround 3.1 / -3.1).
 - Arcsecond " suffix on ticks instead of "y (arcsec)" / "x (arcsec)" labels.
 - Vertical y-tick labels to save horizontal whitespace.
 - A colorbar with large label fontsize.

Since the new API does not expose all matplotlib customization options as kwargs (only the most
common: title, colormap, use_log10, output_path, output_format), we use matplotlib directly after
`aplt.plot_array()` to apply additional customization before showing / saving the figure.

The workflow is:

 1. Call `aplt.plot_array()` to set up the base figure.
 2. Use `plt.gca()` and `plt.gcf()` to access the current axes and figure.
 3. Apply additional matplotlib customization.
 4. Call `plt.savefig()` or `plt.show()`.
"""
aplt.plot_array(array=dataset.data, title="Image — Publication")

ax = plt.gca()

# Large tick fontsize and vertical y-ticks with arcsecond suffix
tick_values = [-2.5, 0.0, 2.5]
ax.set_xticks(tick_values)
ax.set_yticks(tick_values)
ax.set_xticklabels([f'{v}"' for v in tick_values], fontsize=22)
ax.set_yticklabels([f'{v}"' for v in tick_values], fontsize=22, rotation="vertical", va="center")
ax.set_xlabel("")
ax.set_ylabel("")

plt.tight_layout()
plt.show()
plt.close()

"""
__Output__

Save the publication figure to both .png and .pdf.
"""
output_dir = Path("scripts") / "plot" / "publication"
output_dir.mkdir(parents=True, exist_ok=True)

aplt.plot_array(array=dataset.data, title="image_publication")

ax = plt.gca()
ax.set_xticks([-2.5, 0.0, 2.5])
ax.set_yticks([-2.5, 0.0, 2.5])
ax.set_xticklabels(['-2.5"', '0.0"', '2.5"'], fontsize=22)
ax.set_yticklabels(['-2.5"', '0.0"', '2.5"'], fontsize=22, rotation="vertical", va="center")
ax.set_xlabel("")
ax.set_ylabel("")

plt.savefig(output_dir / "image_publication.png", bbox_inches="tight")
plt.savefig(output_dir / "image_publication.pdf", bbox_inches="tight")
plt.close()

"""
__Scale Bar Alternative__

An alternative to showing axis ticks is to draw a scale bar directly on the image,
removing all axis labels and ticks for a cleaner look.
"""
xpos = 2.0
ypos = -2.9

aplt.plot_array(array=dataset.data, title="image_scale_bar")

ax = plt.gca()
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

# Draw a 1.0" scale bar
ax.annotate(
    "",
    xy=(xpos + 1.0, ypos + 0.4),
    xytext=(xpos, ypos + 0.4),
    arrowprops=dict(arrowstyle="-", color="w", lw=2),
)
ax.text(xpos, ypos, '1.0"', color="w", fontsize=30)

plt.tight_layout()
plt.show()
plt.close()
