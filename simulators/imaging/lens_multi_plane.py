import autofit as af
import autolens as al
import autolens.plot as aplt
import os

"""
This tool allows one to make simulated datasets of strong lenses using a multi-plane ray-tracer, such that all
galaxies down the line-of-sight are included in the ray-tracing calculation based on their redshifts.

The 'dataset label' is the name of the dataset folder and 'dataset_name' the folder the dataset is stored in, e.g:

The image will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/image.fits'.
The noise map will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/lens_name/noise_map.fits'.
The psf will be output as '/autolens_workspace/dataset/dataset_label/dataset_name/psf.fits'.
"""

"""Setup the path to the autolens_workspace, using a relative directory name."""
workspace_path = "{}/../../..".format(os.path.dirname(os.path.realpath(__file__)))

"""
The 'dataset_label' describes the type of data being simulated (in this case, imaging data) and 'dataset_name' 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

    - The image will be output to '/autolens_workspace/dataset/dataset_label/dataset_name/image.fits'.
    - The noise map will be output to '/autolens_workspace/dataset/dataset_label/dataset_name/lens_name/noise_map.fits'.
    - The psf will be output to '/autolens_workspace/dataset/dataset_label/dataset_name/psf.fits'.
"""
dataset_label = "imaging"
dataset_name = "lens_multi_plane"

"""
Create the path where the dataset will be output, which in this case is
'/autolens_workspace/dataset/imaging/multi_plane/'
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
grid = al.GridIterate.uniform(
    shape_2d=(100, 100), pixel_scales=0.05, fractional_accuracy=0.9999
)

"""Simulate a simple Gaussian PSF for the image."""
psf = al.Kernel.from_gaussian(
    shape_2d=(11, 11), sigma=0.05, pixel_scales=grid.pixel_scales
)

"""
To simulate the imaging dataset we first create a simulator, which defines the expoosure time, background sky,
noise levels and psf of the dataset that is simulated.
"""
simulator = al.SimulatorImaging(
    exposure_time_map=al.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
    psf=psf,
    background_sky_map=al.Array.full(fill_value=0.1, shape_2d=grid.shape_2d),
    add_noise=True,
)

"""
Setup the lens galaxy's light mass (SIE) and source galaxy light (elliptical Sersic) for this simulated lens.

For lens modeling, defining ellipticity in terms of the  'elliptical_comps' improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle phi, where phi is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoLens** *convert* module to determine the elliptical components from the axis-ratio and phi.
"""

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=45.0),
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    light=al.lp.EllipticalSersic(
        centre=(0.1, 0.1),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=60.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=2.5,
    ),
)

"""
Setup our line-of-sight (los) galaxies using Spherical Sersic profiles for their light and Singular
Isothermal Sphere (SIS) profiles. We'll use 3 galaxies, but you can add more if desired.
"""
los_0 = al.Galaxy(
    redshift=0.25,
    light=al.lp.SphericalSersic(
        centre=(4.0, 4.0), intensity=0.30, effective_radius=0.3, sersic_index=2.0
    ),
    mass=al.mp.SphericalIsothermal(centre=(4.0, 4.0), einstein_radius=0.02),
)

los_1 = al.Galaxy(
    redshift=0.75,
    light=al.lp.SphericalSersic(
        centre=(3.6, -5.3), intensity=0.20, effective_radius=0.6, sersic_index=1.5
    ),
    mass=al.mp.SphericalIsothermal(centre=(3.6, -5.3), einstein_radius=0.04),
)

los_2 = al.Galaxy(
    redshift=1.25,
    light=al.lp.SphericalSersic(
        centre=(-3.1, -2.4), intensity=0.35, effective_radius=0.4, sersic_index=2.5
    ),
    mass=al.mp.SphericalIsothermal(centre=(-3.1, -2.4), einstein_radius=0.03),
)

"""
use these galaxies to setup a multi-plane tracer, which will generate the image for the simulated Imaging
dataset. This tracer orders galaxies by redshift and performs ray-tracing based on their line-of-sight redshifts.
"""
tracer = al.Tracer.from_galaxies(
    galaxies=[lens_galaxy, source_galaxy, los_0, los_1, los_2]
)
aplt.Tracer.image(tracer=tracer, grid=grid)

"""
We can then pass this simulator a tracer, which uses the tracer to create a ray-traced image which is simulated as
imaging dataset following the setup of the dataset.
"""
imaging = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

"""Lets plot the simulated Imaging dataset before we output it to fits."""
aplt.Imaging.subplot_imaging(imaging=imaging)

"""Finally, lets output our simulated dataset to the dataset path as .fits files"""
imaging.output_to_fits(
    image_path=f"{dataset_path}/image.fits",
    psf_path=f"{dataset_path}/psf.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    overwrite=True,
)
