import autolens as al
import autolens.plot as aplt
import numpy as np
import os

workspace_path = "{}/../..".format(os.path.dirname(os.path.realpath(__file__)))

grid = al.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.1, sub_size=1)

y,x = np.unravel_index(abs(grid).argmin(), grid.shape)

print(y, x)

psf = al.Kernel.from_gaussian(
    shape_2d=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

simulator = al.SimulatorImaging(
    psf=psf,
    exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
    background_sky_map=al.Array.full(fill_value=0.1, shape_2d=grid.shape_2d),
    add_noise=True,
)

light_profile = al.lp.EllipticalSersic(
        centre=(1.0, 0.0),
        intensity=1.0,
    )

profile_image = light_profile.profile_image_from_grid(grid=grid)

y,x = np.unravel_index(abs(profile_image.in_2d).argmax(), profile_image.shape_2d)

print(y, x)

aplt.LightProfile.profile_image(light_profile=light_profile, grid=grid)

stop

lens_galaxy = al.Galaxy(
    redshift=0.5,
    light=light_profile,
    # mass=al.mp.EllipticalIsothermal(
    #     centre=(0.0, 0.0), einstein_radius=1.0, axis_ratio=0.999,
    # ),
)
#
# source_galaxy = al.Galaxy(
#     redshift=1.0,
#     light=al.lp.EllipticalSersic(
#         centre=(0.0, 0.0),
#     ),
# )

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy])#, source_galaxy])

aplt.Tracer.profile_image(tracer=tracer, grid=grid)

imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

aplt.Imaging.subplot_imaging(imaging=imaging)