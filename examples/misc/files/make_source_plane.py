"""
This file uses the simulator dataset `imaging/no_lens_light/mass_sie__source_sersic` to create deflection angle map and
image-plane grid.

This is so the `source_planes.py` script can be used to analysis the system in a setting where the deflection angle
map is `known`.
"""

import autolens as al

# %%
"""Use the WORKSPACE environment variable to determine the path to the `autolens_workspace`."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

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

"""The mask and grid of the imaging dataset."""

mask = al.Mask2D.unmasked(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=1
)
grid = al.Grid.from_mask(mask=mask)

"""
The true lens `Galaxy` of the `mass_sie__source_sersic.py` simulator script, which is required to compute the
correct deflection angle map.
"""

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, phi=45.0),
    ),
)

deflections = lens_galaxy.deflections_from_grid(grid=grid)
deflections_y = al.Array.manual_mask(array=deflections.in_1d[:, 0], mask=grid.mask)
deflections_x = al.Array.manual_mask(array=deflections.in_1d[:, 1], mask=grid.mask)

mask.output_to_fits(
    file_path=f"{workspace_path}/examples/misc/files/mask.fits", overwrite=True
)
grid.output_to_fits(
    file_path=f"{workspace_path}/examples/misc/files/grid.fits", overwrite=True
)
deflections_y.output_to_fits(
    file_path=f"{workspace_path}/examples/misc/files/deflections_y.fits", overwrite=True
)
deflections_x.output_to_fits(
    file_path=f"{workspace_path}/examples/misc/files/deflections_x.fits", overwrite=True
)
