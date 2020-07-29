# %%
"""
__Example: Interferometry__

Alongside CCD imaging data, **PyAutoLens** supports the modeling of interferometer data from submm and radio
observatories. Here, the dataset is fitted directly in the uv-plane, circumventing issues that arise when fitting a
'dirty image' such as correlated noise. To begin, we load an interferometer dataset from fits files:
"""

# %%
"""Setup the path to the autolens workspace, using the project pyprojroot which determines it automatically."""

# %%
from pyprojroot import here

workspace_path = str(here())
print("Workspace Path: ", workspace_path)

# %%
"""
Load the strong lens interferometer dataset 'lens_sie__source_sersic' 'from .fits files, which is the dataset we'll use
in this example.
"""

# %%
import autolens as al
import autolens.plot as aplt

dataset_type = "interferometer"
dataset_name = "lens_sie__source_sersic"
dataset_path = f"{workspace_path}/dataset/{dataset_type}/{dataset_name}"

interferometer = al.Interferometer.from_fits(
    visibilities_path=f"{dataset_path}/visibilities.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    uv_wavelengths_path=f"{dataset_path}/uv_wavelengths.fits",
)

aplt.Interferometer.visibilities(interferometer=interferometer)
aplt.Interferometer.uv_wavelengths(interferometer=interferometer)
