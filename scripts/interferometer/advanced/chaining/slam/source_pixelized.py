"""
SLaM (Source, Light and Mass): Mass Total + Source Inversion
============================================================

SLaM pipelines break the analysis of 'galaxy-scale' strong lenses down into multiple pipelines which focus on modeling
a specific aspect of the strong lens, first the Source, then the (lens) Light and finally the Mass. Each of these
pipelines has it own inputs which which customize the model and analysis in that pipeline.

The models fitted in earlier pipelines determine the model used in later pipelines. For example, if the SOURCE PIPELINE
uses a parametric `Sersic` profile for the bulge, this will be used in the subsequent MASS TOTAL PIPELINE.

Using a SOURCE LP PIPELINE, SOURCE PIX PIPELINE and a MASS TOTAL PIPELINE this SLaM script fits `Interferometer` 
of a strong lens system, where in the final model:

 - The lens galaxy's light is omitted from the data and model.
 - The lens galaxy's total mass distribution is an `PowerLaw`.
 - The source galaxy is an `Inversion`.

This uses the SLaM pipelines:

 `source_lp`
 `source__pixelization/source_pixelization__no_lens_light`
 `mass_total`

Check them out for a detailed description of the analysis!

__Run Times and Settings__

The run times of an interferometer `Inversion` depend significantly on the following settings:

 - `transformer_class`: whether a discrete Fourier transform (`TransformerDFT`) or non-uniform fast Fourier Transform
 (`TransformerNUFFT) is used to map the inversion's image from real-space to Fourier space.

 - `use_linear_operators`: whether the linear operator formalism or matrix formalism is used for the linear algebra.

The optimal settings depend on the number of visibilities in the dataset:

 - For N_visibilities < 1000: `transformer_class=TransformerDFT` and `use_linear_operators=False` gives the fastest
 run-times.
 - For  N_visibilities > ~10000: use `transformer_class=TransformerNUFFT`  and `use_linear_operators=True`.

The dataset modeled by default in this script has just 200 visibilties, therefore `transformer_class=TransformerDFT`
and `use_linear_operators=False`.

The script `autolens_workspace/*/interferometer/profiling.py` allows you to compute the run-time of an inversion
for your interferometer dataset. It does this for all possible combinations of settings and therefore can tell you
which settings give the fastest run times for your dataset.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
import sys
from os import path
import autofit as af
import autolens as al
import autolens.plot as aplt

sys.path.insert(0, os.getcwd())
import slam

"""
__Dataset + Masking__ 

Load the `Interferometer` data, define the visibility and real-space masks and plot them.
"""
real_space_mask = al.Mask2D.circular(
    shape_native=(151, 151), pixel_scales=0.05, radius=3.0
)

dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = al.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)
dataset = dataset.apply_settings(
    settings=al.SettingsInterferometer(transformer_class=al.TransformerDFT)
)

"""
__Inversion Settings (Run Times)__

The run times of an interferometer `Inversion` depend significantly on the following settings:

 - `transformer_class`: whether a discrete Fourier transform (`TransformerDFT`) or non-uniform fast Fourier Transform
 (`TransformerNUFFT) is used to map the inversion's image from real-space to Fourier space.

 - `use_linear_operators`: whether the linear operator formalism or matrix formalism is used for the linear algebra.

The optimal settings depend on the number of visibilities in the dataset:

 - For N_visibilities < 1000: `transformer_class=TransformerDFT` and `use_linear_operators=False` gives the fastest
 run-times.
 - For  N_visibilities > ~10000: use `transformer_class=TransformerNUFFT`  and `use_linear_operators=True`.

The dataset modeled by default in this script has just 200 visibilties, therefore `transformer_class=TransformerDFT`
and `use_linear_operators=False`. If you are using this script to model your own dataset with a different number of
visibilities, you should update the options below accordingly.

The script `autolens_workspace/*/interferometer/profiling.py` allows you to compute the run-time of an inversion
for your interferometer dataset. It does this for all possible combinations of settings and therefore can tell you
which settings give the fastest run times for your dataset.
"""
settings_dataset = al.SettingsInterferometer(transformer_class=al.TransformerDFT)
settings_inversion = al.SettingsInversion(use_linear_operators=False)

"""
We now create the `Interferometer` object which is used to fit the lens model.

This includes a `SettingsInterferometer`, which includes the method used to Fourier transform the real-space 
image of the strong lens to the uv-plane and compare directly to the visiblities. We use a non-uniform fast Fourier 
transform, which is the most efficient method for interferometer datasets containing ~1-10 million visibilities.
"""
dataset = dataset.apply_settings(settings=settings_dataset)
dataset_plotter = aplt.InterferometerPlotter(dataset=dataset)
dataset_plotter.subplot_dataset()
dataset_plotter.subplot_dirty_images()

"""
__Settings AutoFit__

The settings of autofit, which controls the output paths, parallelization, database use, etc.
"""
settings_autofit = af.SettingsSearch(
    path_prefix=path.join("interferometer", "slam"),
    unique_tag=dataset_name,
    info=None,
    number_of_cores=1,
    session=None,
)

"""
__Redshifts__

The redshifts of the lens and source galaxies, which are used to perform unit converions of the model and data (e.g. 
from arc-seconds to kiloparsecs, masses to solar masses, etc.).
"""
redshift_lens = 0.5
redshift_source = 1.0

"""
__Adapt Setup__

The `SetupAdapt` determines which hyper-mode features are used during the model-fit.
"""
setup_adapt = al.SetupAdapt(
    mesh_pixels_fixed=1500,
)

"""
__SOURCE LP PIPELINE (no lens light)__

The SOURCE LP PIPELINE (no lens light) uses one search to initialize a robust model for the source galaxy's 
light, which in this example:

 - Uses a parametric `Sersic` bulge for the source's light (omitting a disk / envelope).
 - Uses an `Isothermal` model for the lens's total mass distribution with an `ExternalShear`.
 
__Settings__:
 
 - Mass Centre: Fix the mass profile centre to (0.0, 0.0) (this assumption will be relaxed in the SOURCE INVERSION 
 PIPELINE).
"""
analysis = al.AnalysisInterferometer(dataset=dataset)

source_lp_results = slam.source_lp.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    lens_bulge=None,
    lens_disk=None,
    mass=af.Model(al.mp.Isothermal),
    shear=af.Model(al.mp.ExternalShear),
    source_bulge=af.Model(al.lp.Sersic),
    mass_centre=(0.0, 0.0),
    redshift_lens=0.5,
    redshift_source=1.0,
)

"""
__SOURCE PIX PIPELINE (no lens light)__

The SOURCE PIX PIPELINE (no lens light) uses four searches to initialize a robust model for the `Inversion` that
reconstructs the source galaxy's light. It begins by fitting a `DelaunayMagnification` mesh with `Constant` 
regularization, to set up the model and hyper images, and then:

 - Uses a `DelaunayBrightnessImage` pixelization.
 - Uses an `AdaptiveBrightness` regularization.
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE LP PIPELINE through to the
 SOURCE PIX PIPELINE.

__Settings__:

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    positions_likelihood=source_lp_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=settings_inversion,
)

source_pix_results = slam.source_pix.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_lp_results=source_lp_results,
    mesh=al.mesh.DelaunayBrightnessImage,
    regularization=al.reg.AdaptiveBrightnessSplit,
)

"""
__MASS TOTAL PIPELINE (no lens light)__

The MASS TOTAL PIPELINE (no lens light) uses one search to fits a complex lens mass model to a high level of accuracy, 
using the lens mass model and source model of the SOURCE PIX PIPELINE to initialize the model priors. In this 
example it:

 - Uses an `PowerLaw` model for the lens's total mass distribution [The centre if unfixed from (0.0, 0.0)].
 - Uses an `Inversion` for the source's light [priors fixed from SOURCE PIX PIPELINE].
 - Carries the lens redshift, source redshift and `ExternalShear` of the SOURCE PIPELINES through to the MASS 
 PIPELINE.
 
__Settings__:

 - adapt: We may be using hyper features and therefore pass the result of the SOURCE PIX PIPELINE to use as the
 hyper dataset if required.

 - Positions: We update the positions and positions threshold using the previous model-fitting result (as described 
 in `chaining/examples/parametric_to_pixelization.py`) to remove unphysical solutions from the `Inversion` model-fitting.
"""
analysis = al.AnalysisInterferometer(
    dataset=dataset,
    adapt_result=source_pix_results.last,
    positions_likelihood=source_pix_results.last.positions_likelihood_from(
        factor=3.0, minimum_threshold=0.2
    ),
    settings_inversion=settings_inversion,
)

mass_results = slam.mass_total.run(
    settings_autofit=settings_autofit,
    analysis=analysis,
    setup_adapt=setup_adapt,
    source_results=source_pix_results,
    light_results=None,
    mass=af.Model(al.mp.PowerLaw),
)

"""
Finish.
"""