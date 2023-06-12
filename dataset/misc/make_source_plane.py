"""
This file uses the simulator dataset `imaging/no_lens_light/simple__no_lens_light` to create deflection angle map and
image-plane grid.

This is so the `source_planes.py` script can be used to analysis the system in a setting where the deflection angle
map is `known`.
"""
from os import path
import autolens as al

dataset_type = "imaging"
dataset_label = "no_lens_light"
dataset_name = "simple__no_lens_light"
dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

dataset = al.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
The mask and grid of the imaging dataset.
"""
mask = al.Mask2D.all_false(
    shape_native=imaging.shape_2d, pixel_scales=dataset.pixel_scales
)
grid = al.Grid2D.from_mask(mask=mask)

"""
The true lens `Galaxy` of the `mass_sie__source_lp.py` simulator script, which is required to compute the
correct deflection angle map.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.Isothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.7, angle=45.0),
    ),
)

deflections = lens_galaxy.deflections_yx_2d_from(grid=grid)
deflections_y = al.Array2D(values=deflections.in_1d[:, 0], mask=grid.mask)
deflections_x = al.Array2D(values=deflections.in_1d[:, 1], mask=grid.mask)

output_path = path.join("dataset", "misc")

mask.output_to_fits(file_path=path.join(output_path, "mask.fits"), overwrite=True)
grid.output_to_fits(file_path=path.join(output_path, "grid.fits"), overwrite=True)
deflections_y.output_to_fits(
    file_path=path.join(output_path, "deflections_y.fits"), overwrite=True
)
deflections_x.output_to_fits(
    file_path=path.join(output_path, "deflections_x.fits"), overwrite=True
)
