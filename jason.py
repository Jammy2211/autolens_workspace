from autolens_workspace.howtolens.simulators.chapter_4 import lens_sie__source_sersic

import autolens as al

workspace_path = "/path/to/user/autolens_workspace/howtolens"
workspace_path = "/home/jammy/PycharmProjects/PyAuto/autolens_workspace"

dataset_label = "chapter_4"
dataset_name = "lens_sie__source_sersic"
dataset_path = f"{workspace_path}/howtolens/dataset/{dataset_label}/{dataset_name}"
imaging = al.Imaging.from_fits(
    image_path=f"{dataset_path}/image.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    psf_path=f"{dataset_path}/psf.fits",
    pixel_scales=0.1,
)
mask = al.Mask.circular(
    shape_2d=imaging.shape_2d, pixel_scales=imaging.pixel_scales, sub_size=2, radius=3.0
)
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=1.6
    ),
)
source_galaxy = al.Galaxy(
    redshift=1.0, pixelization=adaptive, regularization=al.reg.Constant(coefficient=1.0)
)
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])
for i in range(4):
    masked_imaging = al.MaskedImaging(
        imaging=imaging, mask=mask, inversion_stochastic=True
    )
    fit = al.FitImaging(masked_imaging=masked_imaging, tracer=tracer)
    print(fit.inversion.mapper.voronoi.points)
