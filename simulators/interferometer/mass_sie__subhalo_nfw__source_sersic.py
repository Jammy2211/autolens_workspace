import autolens as al
import autolens.plot as aplt

"""
This script simulates `Interferometer` data of a strong lens where:

 - The lens `Galaxy`'s total mass distribution is an `EllipticalIsothermal`.
 - The subhalo`s `MassProfile` is a `SphericalNFW`.
 - The source `Galaxy`'s `LightProfile` is an `EllipticalSersic`.
"""

"""
The `dataset_type` describes the type of data being simulated (in this case, `Interferometer` data) and `dataset_name` 
gives it a descriptive name. They define the folder the dataset is output to on your hard-disk:

 - The image will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/image.fits`.
 - The noise-map will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/lens_name/noise_map.fits`.
 - The psf will be output to `/autolens_workspace/dataset/dataset_type/dataset_name/psf.fits`.
"""

dataset_type = "interferometer"
dataset_name = "mass_sie__subhalo_nfw__source_sersic"

"""
The path where the dataset will be output, which in this case is
`/autolens_workspace/dataset/interferometer/mass_sie__subhalo_nfw__source_bulge`
"""

dataset_path = f"dataset/{dataset_type}/{dataset_name}"

"""
For simulating an image of a strong lens, we recommend using a GridIterate object. This represents a grid of (y,x) 
coordinates like an ordinary Grid, but when the light-profile`s image is evaluated below (using the Tracer) the 
sub-size of the grid is iteratively increased (in steps of 2, 4, 8, 16, 24) until the input fractional accuracy of 
99.99% is met.

This ensures that the divergent and bright central regions of the source galaxy are fully resolved when determining the
total flux emitted within a pixel.
"""

grid = al.GridIterate.uniform(
    shape_2d=(151, 151), pixel_scales=0.1, fractional_accuracy=0.9999
)

"""To perform the Fourier transform we need the wavelengths of the baselines, which we'll load from the fits file below."""
uv_wavelengths_path = f"simulators/interferometer/uvtools"

uv_wavelengths = al.util.array.numpy_array_1d_from_fits(
    file_path=f"{uv_wavelengths_path}/sma_uv_wavelengths.fits", hdu=0
)

"""
To simulate the interferometer dataset we first create a simulator, which defines the shape, resolution and pixel-scale 
of the visibilities that are simulated, as well as its exposure time, noise levels and uv-wavelengths.
"""

simulator = al.SimulatorInterferometer(
    uv_wavelengths=uv_wavelengths,
    exposure_time=300.0,
    background_sky_level=0.1,
    noise_sigma=0.1,
    transformer_class=al.TransformerNUFFT,
)


"""
Setup the lens `Galaxy`'s mass (SIE+Shear), subhalo (NFW) and source galaxy light (elliptical Sersic) for this 
simulated lens.

For lens modeling, defining ellipticity in terms of the `elliptical_comps` improves the model-fitting procedure.

However, for simulating a strong lens you may find it more intuitive to define the elliptical geometry using the 
axis-ratio of the profile (axis_ratio = semi-major axis / semi-minor axis = b/a) and position angle phi, where phi is
in degrees and defined counter clockwise from the positive x-axis.

We can use the **PyAutoLens** `convert` module to determine the elliptical components from the axis-ratio and phi.
"""

lens_galaxy = al.Galaxy(
    redshift=0.5,
    mass=al.mp.EllipticalIsothermal(
        centre=(0.0, 0.0),
        einstein_radius=1.6,
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.7, phi=95.0),
    ),
    subhalo=al.mp.SphericalTruncatedNFWMCRLudlow(
        centre=(1.601, 0.0), mass_at_200=1.0e10
    ),
    shear=al.mp.ExternalShear(elliptical_comps=(0.0, 0.05)),
)

source_galaxy = al.Galaxy(
    redshift=1.0,
    bulge=al.lp.EllipticalSersic(
        centre=(0.01, 0.01),
        elliptical_comps=al.convert.elliptical_comps_from(axis_ratio=0.8, phi=60.0),
        intensity=0.3,
        effective_radius=0.3,
        sersic_index=2.5,
    ),
)

"""Use these galaxies to setup a tracer, which will generate the image for the simulated interferometer dataset."""

tracer = al.Tracer.from_galaxies(galaxies=[lens_galaxy, source_galaxy])

"""Lets look at the tracer`s image - this is the image we'll be simulating."""

aplt.Tracer.image(tracer=tracer, grid=grid)

"""
We can now pass this simulator a tracer, which creates the ray-traced image plotted above and simulates it as an
interferometer dataset.
"""
interferometer = simulator.from_tracer_and_grid(tracer=tracer, grid=grid)

"""Lets plot the simulated interferometer dataset before we output it to fits."""

aplt.Interferometer.subplot_interferometer(interferometer=interferometer)

"""Output our simulated dataset to the dataset path as .fits files"""
interferometer.output_to_fits(
    visibilities_path=f"{dataset_path}/visibilities.fits",
    noise_map_path=f"{dataset_path}/noise_map.fits",
    uv_wavelengths_path=f"{dataset_path}/uv_wavelengths.fits",
    overwrite=True,
)
