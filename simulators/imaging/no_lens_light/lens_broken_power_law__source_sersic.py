import autofit as af
import autolens as al
import autolens.plot as aplt
import os

"""
This script simulates imaging of a strong lens where:

    - The lens galaxy's _MassProfile_ is an EllipticalBrokenPowerLaw.
    - The source galaxy's _LightProfile_ is an _EllipticalSersic_.
"""

"""Setup the path to the autolens_workspace, using a relative directory name."""
workspace_path = "{}/../../..".format(os.path.dirname(os.path.realpath(__file__)))

"""
The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

    - The image will be output to '/autolens_workspace/dataset/dataset_label/dataset_name/image.fits'.
    - The noise-map will be output to '/autolens_workspace/dataset/dataset_label/dataset_name/lens_name/noise_map.fits'.
    - The psf will be output to '/autolens_workspace/dataset/dataset_label/dataset_name/psf.fits'.
"""
dataset_label = "imaging"
dataset_name = "lens_broken_power_law__source_sersic"

"""
Create the path where the dataset will be output, which in this case is:
'/autolens_workspace/dataset/imaging/lens_broken_power_law__source_sersic/'
"""
dataset_path = af.util.create_path(
    path=workspace_path, folders=["dataset", dataset_label, dataset_name]
)

"""
The grid used to simulate the image. 

For simulating an image of a strong lens, we recommend using a GridIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid, but when the light-profile's image is evaluated below (using the Tracer) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the source galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""
grid = al.GridIterate.uniform(shape_2d=(180, 180), pixel_scales=0.05)

"""Simulate a simple Gaussian PSF for the image."""
psf = al.Kernel.from_gaussian(
    shape_2d=(11, 11), sigma=0.1, pixel_scales=grid.pixel_scales
)

"""
To simulate the imaging dataset we first create a simulator, which defines the expoosure time, psf and noise levels of
the simulated image.
"""
simulator = al.SimulatorImaging(
    exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
    psf=psf,
    background_sky_map=al.Array.full(fill_value=0.1, shape_2d=grid.shape_2d),
    add_noise=True,
)

"""
Setup the lens galaxy's mass (SIE+Shear) and source galaxy light (elliptical Sersic) for this simulated lens.

For lens modeling, defining ellipticity in terms of the  'elliptical_comps' improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle phi, where phi is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoLens** *convert* module to determine the elliptical components from the axis-ratio and phi.
"""
lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalBrokenPowerLaw(
        centre=(0.0, 0.0),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.3, phi=45.0),
        einstein_radius=1.3,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.5,
    ),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.03, 0.03),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=60.0),
        intensity=0.3,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

"""Use these galaxies to setup a tracer, which will generate the image for the simulated imaging dataset."""
tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""Lets look at the tracer's image - this is the image we'll be simulating."""
aplt.Tracer.image(tracer=tracer, grid=grid)
aplt.Tracer.subplot_tracer(tracer=tracer, grid=grid)

"""
We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
imaging dataset.
"""
imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

"""Lets plot the simulated imaging dataset before we output it to fits."""
aplt.Imaging.subplot_imaging(imaging=imaging)

"""Finally, lets output our simulated dataset to the dataset path as .fits files"""
imaging.output_to_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    overwrite=True,
)
