"""
Simulator: Wavelength Dependent
===============================

This script simulates multi-wavelength `Imaging` of a 'galaxy-scale' strong lens where:

 - The lens galaxy's light is a parametric `Sersic` bulge where the `intensity` varies across wavelength.
 - The lens galaxy's total mass distribution is an `Isothermal` and `ExternalShear`.
 - The source galaxy's light is an `Sersic`, which has a different `intensity` at each wavelength.

Unlike other `multi` simulators, the intensity of the source galaxy is a linear function of wavelength following
a relation `y = mx + c`.

This image is used to demonstrate multi-wavelength fitting where a user specified function (e.g. `y = mx+c`) can be
used to parameterize the wavelength variation, as opposed to simply making every `intensity` a free parameter.

Three images are simulated, corresponding green g band (wavelength=464nm), red r-band (wavelength=658nm) and
infrared I-band (wavelength=806nm) observations.

This is an advanced script and assumes previous knowledge of the core **PyAutoLens** API for simulating images. Thus,
certain parts of code are not documented to ensure the script is concise.
"""

# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autolens as al
import autolens.plot as aplt

"""
__Colors__

The colors of the multi-wavelength image, which in this case are green (g-band), red (r-band) and infrared (I-band).

The strings are used for naming the datasets on output.
"""
color_list = ["g", "r", "I"]

"""
__Wavelengths__

The intensity of each source galaxy is parameterized as a function of wavelength.

Therefore we define a list of wavelengths of each color above.
"""
wavelength_list = [464, 658, 806]

"""
__Dataset Paths__
"""
dataset_type = "multi"
dataset_label = "imaging"
dataset_name = "wavelength_dependence"

dataset_path = path.join("dataset", dataset_type, dataset_label, dataset_name)

"""
__Simulate__

The pixel-scale of each color image is different meaning we make a list of grids for the simulation.
"""
pixel_scales_list = [0.08, 0.12, 0.012]

grid_list = []

for pixel_scales in pixel_scales_list:
    grid = al.Grid2D.uniform(
        shape_native=(150, 150),
        pixel_scales=pixel_scales,
    )

    over_sample_size = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=grid,
        sub_size_list=[32, 8, 2],
        radial_list=[0.3, 0.6],
        centre_list=[(0.0, 0.0)],
    )

    grid = grid.apply_over_sampling(over_sample_size=over_sample_size)

    grid_list.append(grid)


"""
Simulate simple Gaussian PSFs for the images in the r and g bands.
"""
sigma_list = [0.1, 0.2, 0.25]

psf_list = [
    al.Kernel2D.from_gaussian(
        shape_native=(11, 11), sigma=sigma, pixel_scales=grid.pixel_scales
    )
    for grid, sigma in zip(grid_list, sigma_list)
]

"""
Create separate simulators for the g and r bands.
"""
background_sky_level_list = [0.1, 0.15, 0.1]

simulator_list = [
    al.SimulatorImaging(
        exposure_time=300.0,
        psf=psf,
        background_sky_level=background_sky_level,
        add_poisson_noise_to_data=True,
    )
    for psf, background_sky_level in zip(psf_list, background_sky_level_list)
]

"""
__Intensity vs Wavelength__

We will assume that the `intensity` of the lens and source galaxies linearly varies as a function of wavelength, and 
therefore compute the `intensity` value for each color image using a linear relation.

The relation below is not realistic and has been chosen to make it straight forward to illustrate this functionality.
"""


def lens_intensity_from(wavelength):
    m = 1.0 / 100.0
    c = 3

    return m * wavelength + c


def source_intensity_from(wavelength):
    m = -(1.2 / 100.0)
    c = 10

    return m * wavelength + c


"""
__Ray Tracing__

Setup the lens galaxy's mass (SIE+Shear) for this simulated lens.
"""
lens_intensity_list = [
    lens_intensity_from(wavelength=wavelength) for wavelength in wavelength_list
]

bulge_list = [
    al.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=intensity,
        effective_radius=0.8,
        sersic_index=4.0,
    )
    for intensity in lens_intensity_list
]

mass = al.mp.Isothermal(
    centre=(0.0, 0.0),
    einstein_radius=1.6,
    ell_comps=al.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
)

lens_galaxy_list = [
    al.Galaxy(
        redshift=0.5,
        bulge=bulge,
        mass=mass,
        shear=al.mp.ExternalShear(gamma_1=0.05, gamma_2=0.05),
    )
    for bulge in bulge_list
]

"""
__Intensity vs Wavelength__

We will assume that the `intensity` of the source galaxy linearly varies as a function of wavelength, and therefore
compute the `intensity` value for each color image using a linear relation.

The relation below is not realistic and has been chosen to make it straight forward to illustrate this functionality.
"""
source_intensity_list = [
    source_intensity_from(wavelength=wavelength) for wavelength in wavelength_list
]

"""
The source galaxy at each wavelength has a different intensity, thus we create three source galaxies for each waveband.
"""
source_galaxy_list = [
    al.Galaxy(
        redshift=1.0,
        bulge=al.lp.SersicCore(
            centre=(0.0, 0.0),
            ell_comps=al.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
            intensity=intensity,
            effective_radius=0.1,
            sersic_index=1.0,
        ),
    )
    for intensity in source_intensity_list
]

"""
Use these galaxies to setup tracers at each waveband, which will generate each image for the simulated `Imaging` 
dataset.
"""
tracer_list = [
    al.Tracer(galaxies=[lens_galaxy, source_galaxy])
    for lens_galaxy, source_galaxy in zip(lens_galaxy_list, source_galaxy_list)
]

"""
Lets look at the tracer`s image, this is the image we'll be simulating.
"""
for tracer, grid in zip(tracer_list, grid_list):
    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid)
    tracer_plotter.figures_2d(image=True)

"""
Pass the simulator a tracer, which creates the image which is simulated as an imaging dataset.
"""
dataset_list = [
    simulator.via_tracer_from(tracer=tracer, grid=grid)
    for grid, simulator, tracer in zip(grid_list, simulator_list, tracer_list)
]

"""
Plot the simulated `Imaging` dataset before outputting it to fits.
"""
for dataset in dataset_list:
    dataset_plotter = aplt.ImagingPlotter(dataset=dataset)
    dataset_plotter.subplot_dataset()

"""
__Output__

Output each simulated dataset to the dataset path as .fits files, with a tag describing its color.
"""
for color, dataset in zip(color_list, dataset_list):
    dataset.output_to_fits(
        data_path=path.join(dataset_path, f"{color}_data.fits"),
        psf_path=path.join(dataset_path, f"{color}_psf.fits"),
        noise_map_path=path.join(dataset_path, f"{color}_noise_map.fits"),
        overwrite=True,
    )

"""
__Visualize__

Output a subplot of the simulated dataset, the image and the tracer's quantities to the dataset path as .png files.
"""
for color, dataset in zip(color_list, dataset_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{color}_", format="png")
    )

    dataset_plotter = aplt.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    dataset_plotter.subplot_dataset()
    dataset_plotter.figures_2d(data=True)

for color, grid, tracer in zip(color_list, grid_list, tracer_list):
    mat_plot = aplt.MatPlot2D(
        output=aplt.Output(path=dataset_path, prefix=f"{color}_", format="png")
    )

    tracer_plotter = aplt.TracerPlotter(tracer=tracer, grid=grid, mat_plot_2d=mat_plot)
    tracer_plotter.subplot_tracer()
    tracer_plotter.subplot_galaxies_images()

"""
__Tracer json__

Save the `Tracer` in the dataset folder as a .json file, ensuring the true light profiles, mass profiles and galaxies
are safely stored and available to check how the dataset was simulated in the future. 

This can be loaded via the method `tracer = al.from_json()`.
"""
[
    al.output_to_json(
        obj=tracer, file_path=path.join(dataset_path, f"{color}_tracer.json")
    )
    for color, tracer in zip(color_list, tracer_list)
]

"""
The dataset can be viewed in the folder `autolens_workspace/imaging/multi/simple__no_lens_light`.
"""
