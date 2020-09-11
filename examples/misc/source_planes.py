"""
__Example: Source Planes__

In this example, we use a deflection angle map from an external source to create and investigate the source-plane of an
_Imaging_ dataset.

Therefore, this example is for a problem where you have a 'known' deflection angle map from outside PyAutoLens
(e.g. a model computed by another code) and wish to investigate it with PyAutoLens. An obvious example might be
the deflection angle map of a strong lensing galaxy cluster.
"""

# %%
"""Use the WORKSPACE environment variable to determine the path to the autolens workspace."""

# %%
import os

workspace_path = os.environ["WORKSPACE"]
print("Workspace Path: ", workspace_path)

import autolens as al
import autolens.plot as aplt
import numpy as np

"""
In this example, our 'known' deflection angle comes from the _Imaging_ data of a strong lens, specifically that 
created in the 'mass_sie__source_sersic.py' simulator. 

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
In 'autolens_workspace/examples/misc/files' you'll find the script 'make_source_plane.py', which creates the _Grid_ 
and deflection angles we use in this example (using the same _Galaxy_'s as the 'mass_sie__source_sersic.py' simulator). 
"""

"""Lets load the 'known' deflection angle map from a .fits files (which is created in the code mentioned above)."""
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

"""
We now apply a mask to the above image-plane _Grid_ and deflection angles, removing areas where the image 
is not located.

In this example we'll use a 3.0" circular _Mask_.
"""

mask = al.Mask.circular(
    shape_2d=grid.shape_2d, pixel_scales=imaging.pixel_scales, radius=3.0
)

"""The manual_mask function essentially takes the 2D image of each deflection angle map and applies the masked."""

deflections_y = al.Array.manual_mask(array=deflections_y.in_2d, mask=mask)
deflections_x = al.Array.manual_mask(array=deflections_x.in_2d, mask=mask)
aplt.Array(array=deflections_y)
aplt.Array(array=deflections_x)

"""We can use the same function to masks the image-plane _Grid_."""
grid = al.Grid.manual_mask(grid=grid.in_2d, mask=mask)
aplt.Grid(grid=grid)

"""We also apply this mask to our _Imaging_ data, for when we fit it below."""
masked_imaging = al.MaskedImaging(imaging=imaging, mask=mask)

"""
Now we have the image-plane _Grid_ and deflection angles, we can subtract the latter former from the latter to
create our source-plane grid.

(PyAutoLens formats required us to pass in the delfections as a stacked ndarray of the (y,x) components).
"""
deflections = np.stack((deflections_y, deflections_x), axis=-1)
source_plane_grid = grid.grid_from_deflection_grid(deflection_grid=deflections)
aplt.Grid(grid=source_plane_grid)

"""
We are now about to make a source _Galaxy_, place it on this source-plane _Grid_ and create an image of it.
"""
source_galaxy = al.Galaxy(
    redshift=1.0,
    sersic=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)
source_plane = al.Plane(galaxies=[source_galaxy])
aplt.Plane.plane_image(
    plane=source_plane,
    grid=source_plane_grid,
    #   include=aplt.Include(grid=True) # You can inclue the _Grid_ by uncommented this, but it blocks the source.
)

"""
The source-plane _Grid_ and _Plane_ can be used to create the source's _Galaxy_'s lensed image in the image-plane.
"""
image_plane_image = source_plane.image_from_grid(grid=source_plane_grid)
aplt.Plane.image(plane=source_plane, grid=source_plane_grid)

"""
Now, we can ask a crucial question - how well does the source _Galaxy_ used above in combination with our known
deflection angle map fit the image of a strong lens we are comparing to?

To do this, we can manually subtract the image above from the data to create residuals and compare.
"""
residual_map = masked_imaging.image - image_plane_image
aplt.Array(array=residual_map)

"""
It gives a good fit, however there are noticeable residuals are because we did not blur the image_plane_image with 
the _Imaging_ data's PSF first. Lets address this omission.
"""
blurred_image_plane_image = masked_imaging.psf.convolved_array_from_array(
    array=image_plane_image
)
residual_map = masked_imaging.image - blurred_image_plane_image
aplt.Array(array=residual_map)

"""
We can also use a _Pixelization_ and _Regularization_ (which combined create an _Inversion_) to reconstruct the
source galaxy.

We'll reconstruct the source on a 30 x 30 _Rectangular_ source-plane _Pixelization_.
"""

pixelization = al.pix.Rectangular(shape=(30, 30))

"""
A _Mapper_ maps the source-pixels to image-pixels, as shown in the figure below. These mappings are used when 
reconstructing the source _Galaxy_'s light.
"""

mapper = pixelization.mapper_from_grid_and_sparse_grid(grid=source_plane_grid)

aplt.Mapper.subplot_image_and_mapper(
    image=imaging.image,
    mapper=mapper,
    include=aplt.Include(grid=True),
    source_pixel_indexes=[[312], [314], [350], [370]],
)

"""
We can now use a _Mapper_ to perform the _Inversion_ and reconstruct the source _Galaxy_'s light. 

To perform this _Inverison_ we must also input a _Regularization_, which is a prior on how much we smooth the 
source _Galaxy_'s light. Try increasing / decreasing the coefficient value to see what effect this has.
"""

regularization = al.reg.Constant(coefficient=1.0)

inversion = al.Inversion(
    masked_dataset=masked_imaging,
    mapper=mapper,
    regularization=al.reg.Constant(coefficient=1.0),
)

"""
Finally, lets plot: 
 - The reconstruction of the source _Galaxy- in the source-plane.
 - The corresponding reconstructed image-plane image of the lensed source _Galaxy_ (which accounts for PSF blurring).
 - The residuals of the fit to the _MaskedImaging_.
"""

aplt.Inversion.reconstruction(inversion=inversion)
aplt.Inversion.reconstructed_image(inversion=inversion)
residual_map = masked_imaging.image - inversion.mapped_reconstructed_image
aplt.Array(array=residual_map)
